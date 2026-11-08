"""
Federated-learning aggregation schemes behind one interface.

    Scheme A  SequentialAggregator     one process, in-place weighted sum
    Scheme B  PickleAggregator         persistent worker pool, updates pickled
                                       and copied through pipes (partition by clients)
    Scheme C  SharedMemoryAggregator   persistent worker pool, updates live in one
                                       multiprocessing.shared_memory block, workers
                                       attach zero-copy views (partition by model
                                       slice or by clients)
    Optional  NumbaAggregator          in-process multithreaded kernel (Khan et al.)

Every scheme computes  global = sum_c w[c] * U[c]  with normalized weights, using
the same inner NumPy kernel (accumulate()), so measured differences come from the
mechanism (parallelism / data movement), not from different arithmetic.
"""
import os
import sys

# Keep BLAS single-threaded BEFORE numpy is imported, otherwise the "sequential"
# baseline could secretly use several cores (and workers would oversubscribe).
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gc
import pickle
import time
import threading
from threading import BrokenBarrierError
import traceback
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

DTYPE = np.float32
ITEM = np.dtype(DTYPE).itemsize
# Set finite barrier timeout to prevent hangs on all platforms
BARRIER_TIMEOUT_S = 900
ALIGN = 16  # slice boundaries aligned to 16 floats = 64 B cache line (no false sharing)


# --------------------------------------------------------------------------- kernel
# Inner tile, in float32 elements, for the cache-blocked kernel.
#
# Each concurrent worker keeps two private tiles hot: the accumulator slice of
# `out` and the scratch buffer `tmp`. A single worker is therefore happy with a
# 1 MB tile, but W workers need W x 2 x tile bytes of last-level cache between
# them. With the historical fixed 1 MB tile, eight workers need 16 MB, which is
# exactly the LLC of the machine used here, and measured throughput collapses
# (see the tile-size ablation in the paper). The tile is therefore derived from
# the concurrency: we budget a quarter of the LLC for the workers' private
# tiles, i.e. W * 2 * tile <= LLC / 4.
#
# Tiling never changes the arithmetic: within a tile each output element still
# accumulates clients in order 0..n-1, so results are independent of tile size
# and slice partitioning stays bit-identical to the sequential baseline.
MAX_TILE = 262_144           # 1 MB of float32, the historical fixed value
MIN_TILE = 16_384            # 64 KB of float32
DEFAULT_LLC = 8 * 1024 ** 2  # assumed last-level cache when detection fails


def llc_bytes():
    """Size of the last-level cache in bytes, or DEFAULT_LLC if unknown."""
    try:
        if sys.platform.startswith("linux"):
            best = 0
            base = "/sys/devices/system/cpu/cpu0/cache"
            for idx in os.listdir(base):
                if not idx.startswith("index"):
                    continue
                try:
                    with open(os.path.join(base, idx, "size")) as f:
                        s = f.read().strip()
                except OSError:
                    continue
                mult = {"K": 1024, "M": 1024 ** 2}.get(s[-1].upper(), 1)
                best = max(best, int(s.rstrip("KMkm")) * mult)
            return best or DEFAULT_LLC
        if sys.platform == "darwin":
            import subprocess
            for key in ("hw.l3cachesize", "hw.l2cachesize"):
                try:
                    v = int(subprocess.check_output(["sysctl", "-n", key], text=True))
                    if v:
                        return v
                except Exception:
                    continue
            return DEFAULT_LLC
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            RelationCache = 2
            k32 = ctypes.windll.kernel32
            need = wintypes.DWORD(0)
            k32.GetLogicalProcessorInformationEx(RelationCache, None,
                                                 ctypes.byref(need))
            buf = (ctypes.c_ubyte * need.value)()
            if not k32.GetLogicalProcessorInformationEx(RelationCache, buf,
                                                        ctypes.byref(need)):
                return DEFAULT_LLC
            best, off = 0, 0
            while off < need.value:
                # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship, Size,
                # then CACHE_RELATIONSHIP { Level, Assoc, LineSize, CacheSize }
                size = int.from_bytes(bytes(buf[off + 4:off + 8]), "little")
                if size == 0:
                    break
                cache_size = int.from_bytes(bytes(buf[off + 12:off + 16]), "little")
                best = max(best, cache_size)
                off += size
            return best or DEFAULT_LLC
    except Exception:
        pass
    return DEFAULT_LLC


_LLC = llc_bytes()


def tile_elems(W, m=None):
    """Inner tile in float32 elements for W concurrent workers over m params.

    Budget a quarter of the LLC for the W workers' two private tiles each, clamp
    into [MIN_TILE, MAX_TILE], and never use a tile larger than the work itself
    (so small models keep running as a single untiled pass, as before).
    An explicit FL_AGG_TILE_KB overrides the rule, for the ablation.
    """
    env = os.environ.get("FL_AGG_TILE_KB")
    if env:
        t = max(1024, int(float(env) * 1024) // ITEM)
    else:
        t = max(MIN_TILE, min(MAX_TILE, _LLC // (8 * max(1, W) * ITEM)))
    return max(1, min(t, m) if m else t)


# Backwards-compatible single-worker tile.
BLOCK = tile_elems(1)


def accumulate(U, w, out, tmp, kernel="blocked", block=None):
    """out += sum_c w[c] * U[c]   (U: (n, m) view, w: (n,), out/tmp: (m,) float32).

    blocked : default. Walk the parameters in tiles sized by tile_elems() and,
              inside each tile, loop over clients in order. Every update byte is
              read from DRAM once and tmp/out stay in cache, so the sequential
              baseline is as cache-friendly as a parallel slice (no cache
              artifact in speedups).
    axpy    : literal per-client loop of the pseudocode over the full vector
              (tmp/out spill out of cache for large models -> ~5 memory passes).
    gemv    : one single-threaded BLAS matrix-vector product.
    Per element, clients are always summed in order 0..n-1 (blocked/axpy), so
    the result does not depend on `block`.
    """
    n, m = U.shape
    if n == 0 or m == 0:
        return out
    if kernel == "gemv":
        np.dot(w, U, out=tmp)
        np.add(out, tmp, out=out)
        return out
    step = (block or BLOCK) if kernel == "blocked" else m
    for a in range(0, m, step):
        b = min(m, a + step)
        o, t = out[a:b], tmp[:b - a]
        for c in range(n):
            np.multiply(U[c, a:b], w[c], out=t)
            np.add(o, t, out=o)
    return out


def slice_bounds(P, W, k):
    """Contiguous slice [a, b) of the P parameters for worker k (cache-line aligned)."""
    step = -(-P // W)
    step = -(-step // ALIGN) * ALIGN
    a = min(P, k * step)
    return a, min(P, a + step)


def client_bounds(N, W, k):
    """Disjoint client range [lo, hi) for worker k."""
    return (k * N) // W, ((k + 1) * N) // W


def default_start_method():
    # Methodology: Linux -> fork, Windows -> spawn. macOS also uses spawn.
    return "fork" if sys.platform.startswith("linux") else "spawn"


# --------------------------------------------------------------- shared-memory utils
def _attach(name):
    """Attach to an existing block without registering it with the POSIX resource
    tracker (otherwise worker exit can unlink the parent's block / spam warnings)."""
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    if os.name == "nt":
        return shared_memory.SharedMemory(name=name)
    # NOTE: this temporarily replaces resource_tracker.register process-wide so the
    # attached segment is not registered for cleanup by this process. It is only safe
    # if no other thread attaches shared memory concurrently, since they would see
    # the patched function. The original is restored in the finally block below.
    from multiprocessing import resource_tracker
    orig = resource_tracker.register
    resource_tracker.register = lambda *a, **k: None
    try:
        return shared_memory.SharedMemory(name=name)
    finally:
        resource_tracker.register = orig


def _close_shm(shm, unlink=False):
    if shm is None:
        return
    for _ in range(3):
        try:
            shm.close()
            break
        except BufferError:  # a numpy view still exists somewhere
            gc.collect()
    if unlink:
        try:
            shm.unlink()  # no-op on Windows (freed when the last handle closes)
        except FileNotFoundError:
            pass


# ------------------------------------------------------------------- worker process
class _WorkerState:
    def __init__(self, kernel, W=1):
        self.kernel = kernel
        self.W = W
        self.shms = {}           # name -> SharedMemory (cached attachments)
        self.tmp = np.empty(0, DTYPE)
        self.partial = None      # Scheme B partial sum

    def get_tmp(self, m):
        if self.tmp.size < m:
            self.tmp = np.empty(m, DTYPE)
        return self.tmp[:m]

    def shm(self, name):
        s = self.shms.get(name)
        if s is None:
            s = _attach(name)
            self.shms[name] = s
        return s

    def release(self):
        for s in self.shms.values():
            _close_shm(s)
        self.shms.clear()


def _run_shared(st, msg):
    """Scheme C work item. All arrays are local so no view outlives the call."""
    _, in_name, out_name, part_name, N, P, mode, k, W = msg
    ibuf = st.shm(in_name).buf
    U = np.ndarray((N, P), DTYPE, buffer=ibuf)
    w = np.ndarray((N,), DTYPE, buffer=ibuf, offset=N * P * ITEM)
    if mode == "slice":
        a, b = slice_bounds(P, W, k)
        if b > a:
            out = np.ndarray((P,), DTYPE, buffer=st.shm(out_name).buf)[a:b]
            out[...] = 0
            accumulate(U[:, a:b], w, out, st.get_tmp(b - a), st.kernel,
                       tile_elems(W, b - a))
    else:  # partition by clients -> partial vector per worker, parent reduces
        lo, hi = client_bounds(N, W, k)
        part = np.ndarray((W, P), DTYPE, buffer=st.shm(part_name).buf)[k]
        part[...] = 0
        accumulate(U[lo:hi], w[lo:hi], part, st.get_tmp(P), st.kernel,
                   tile_elems(W, P))


def _worker_main(k, conn, barrier, kernel, W=1):
    st = _WorkerState(kernel, W)
    conn.send(("ready", os.getpid()))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        tag = msg[0]
        try:
            if tag == "C_run":
                err = None
                try:
                    _run_shared(st, msg)
                except Exception:
                    err = traceback.format_exc()
                if err is not None:
                    conn.send(("error", err))
                barrier.wait(BARRIER_TIMEOUT_S)  # signal: my slice is written
            elif tag == "B_begin":
                P = msg[1]
                st.partial = np.zeros(P, DTYPE)
            elif tag == "B_batch":
                _, rows, wts = msg       # already unpickled by conn.recv()
                accumulate(rows, wts, st.partial, st.get_tmp(st.partial.size),
                           st.kernel, tile_elems(st.W, st.partial.size))
                del rows, wts
            elif tag == "B_end":
                conn.send(("B_result", st.partial))   # pickled + copied back
                st.partial = None
            elif tag == "release":
                st.release()
                conn.send(("released",))
            elif tag == "ping":
                conn.send(("pong",))
            elif tag == "stop":
                st.release()
                break
        except Exception:
            try:
                conn.send(("error", traceback.format_exc()))
            except Exception:
                pass
    conn.close()


class WorkerPool:
    """W persistent worker processes, created once and reused across rounds."""

    def __init__(self, W, start_method=None, kernel="blocked"):
        self.W = W
        self.start_method = start_method or default_start_method()
        ctx = mp.get_context(self.start_method)
        t0 = time.perf_counter()
        self.barrier = ctx.Barrier(W + 1)
        self.conns, self.procs = [], []
        for k in range(W):
            parent, child = ctx.Pipe(duplex=True)
            p = ctx.Process(target=_worker_main, args=(k, child, self.barrier, kernel, W),
                            daemon=True, name="agg-worker-%d" % k)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.pids = []
        for c in self.conns:
            tag, pid = c.recv()
            self.pids.append(pid)
        self.startup_s = time.perf_counter() - t0  # reported separately, never in a round

    def check_errors(self):
        for c in self.conns:
            if c.poll():
                msg = c.recv()
                if msg[0] == "error":
                    raise RuntimeError("worker failed:\n" + msg[1])

    def release_shared(self):
        for c in self.conns:
            c.send(("release",))
        for c in self.conns:
            msg = c.recv()
            if msg[0] == "error":
                raise RuntimeError("worker failed:\n" + msg[1])

    def close(self):
        for c in self.conns:
            try:
                c.send(("stop",))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        for c in self.conns:
            c.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- schemes
class SequentialAggregator:
    """Scheme A: one process, one core, no IPC."""
    name = "A"

    def __init__(self, kernel="blocked"):
        self.kernel = kernel
        self.out = self.tmp = None

    def aggregate(self, U, w):
        """Aggregate client updates sequentially on one core.
        WARNING: the returned array is an internal buffer (self.out) that is
        overwritten on the next call. Call .copy() if you need to keep it.
        """
        P = U.shape[1]
        if self.out is None or self.out.size != P:
            self.out, self.tmp = np.empty(P, DTYPE), np.empty(P, DTYPE)
        self.out[...] = 0
        return accumulate(U, w, self.out, self.tmp, self.kernel,
                          tile_elems(1, P))

    def close(self):
        pass


class PickleAggregator:
    """Scheme B: clients partitioned across W workers; each worker's chunk of full
    update tensors is explicitly pickled and copied through its pipe. Chunks are
    streamed in batches of <= batch_mb so a 3.5 GB block does not need 3x RAM;
    every byte of the N*P updates is still serialized and copied exactly once."""
    name = "B"

    def __init__(self, pool, batch_mb=64):
        self.pool = pool
        self.batch_bytes = int(batch_mb * 2 ** 20)
        self.out = None

    def aggregate(self, U, w):
        """Aggregate client updates via multi-process pickling.
        WARNING: the returned array is an internal buffer (self.out) that is
        overwritten on the next call. Call .copy() if you need to keep it.
        """
        N, P = U.shape
        W, conns = self.pool.W, self.pool.conns
        rows_per_batch = max(1, self.batch_bytes // (P * ITEM))
        bounds = [client_bounds(N, W, k) for k in range(W)]
        cursor = [lo for lo, _ in bounds]
        for c in conns:
            c.send(("B_begin", P))
        pending = True
        while pending:                       # round-robin so workers compute while
            pending = False                  # the parent pickles the next batch
            for k, c in enumerate(conns):
                lo, hi = cursor[k], bounds[k][1]
                if lo >= hi:
                    continue
                b = min(hi, lo + rows_per_batch)
                payload = pickle.dumps(("B_batch", U[lo:b], w[lo:b]),
                                       protocol=pickle.HIGHEST_PROTOCOL)  # full copy
                c.send_bytes(payload)
                del payload
                cursor[k] = b
                pending = pending or b < hi
        for c in conns:
            c.send(("B_end",))
        if self.out is None or self.out.size != P:
            self.out = np.empty(P, DTYPE)
        self.out[...] = 0
        for c in conns:
            msg = c.recv()                   # unpickle partial vector
            if msg[0] == "error":
                raise RuntimeError("worker failed:\n" + msg[1])
            np.add(self.out, msg[1], out=self.out)
        return self.out

    def close(self):
        pass


class SharedUpdateBlock:
    """One multiprocessing.shared_memory block holding all N updates (row-major,
    float32) followed by the N normalized weights. Updates can be generated or
    received straight into .U, so no duplicate copy is ever held."""

    def __init__(self, N, P):
        self.N, self.P = N, P
        self.shm = shared_memory.SharedMemory(create=True, size=(N * P + N) * ITEM)
        self.name = self.shm.name
        self.U = np.ndarray((N, P), DTYPE, buffer=self.shm.buf)
        self.w = np.ndarray((N,), DTYPE, buffer=self.shm.buf, offset=N * P * ITEM)

    @property
    def nbytes(self):
        return (self.N * self.P + self.N) * ITEM

    def close(self):
        self.U = self.w = None
        gc.collect()
        _close_shm(self.shm, unlink=True)
        self.shm = None


class SharedMemoryAggregator:
    """Scheme C: workers attach to the shared update block as zero-copy views and
    write their result straight into a shared output buffer. Only metadata
    (block names, shape, worker index) is pickled and sent per round."""
    name = "C"

    def __init__(self, pool, partition="slice"):
        assert partition in ("slice", "clients")
        self.pool = pool
        self.partition = partition
        self.block = None
        self._own_block = False
        self.shm_out = self.shm_part = None
        self.out = self.partials = self._red = None

    def use_block(self, block, own=False):
        """Point this aggregator at an existing SharedUpdateBlock."""
        if self.block is block:
            return
        self.release()
        self.block, self._own_block = block, own
        P, W = block.P, self.pool.W
        self.shm_out = shared_memory.SharedMemory(create=True, size=P * ITEM)
        self.out = np.ndarray((P,), DTYPE, buffer=self.shm_out.buf)
        if self.partition == "clients":
            self.shm_part = shared_memory.SharedMemory(create=True, size=W * P * ITEM)
            self.partials = np.ndarray((W, P), DTYPE, buffer=self.shm_part.buf)
            self._red = np.empty(P, DTYPE)

    def release(self):
        """Workers detach, then the parent releases its blocks."""
        if self.block is None:
            return
        self.pool.release_shared()
        self.out = self.partials = self._red = None
        gc.collect()
        _close_shm(self.shm_out, unlink=True)
        _close_shm(self.shm_part, unlink=True)
        self.shm_out = self.shm_part = None
        if self._own_block:
            self.block.close()
        self.block, self._own_block = None, False

    POLL_S = 0.5

    def _watchdog(self, stop: threading.Event):
        while not stop.wait(self.POLL_S):
            dead = any(not p.is_alive() for p in self.pool.procs)
            errored = any(c.poll() for c in self.pool.conns)
            if dead or errored:
                self.pool.barrier.abort()
                return

    def aggregate_resident(self):
        """Compute-only: the updates are already resident in the shared block.
        WARNING: the returned array is an internal buffer (self.out) that is
        overwritten on the next call. Call .copy() if you need to keep it.
        """
        b, W = self.block, self.pool.W
        part_name = self.shm_part.name if self.shm_part is not None else None
        for k, c in enumerate(self.pool.conns):
            c.send(("C_run", b.name, self.shm_out.name, part_name,
                    b.N, b.P, self.partition, k, W))
        stop = threading.Event()
        wd = threading.Thread(target=self._watchdog, args=(stop,), daemon=True)
        wd.start()
        try:
            self.pool.barrier.wait(timeout=BARRIER_TIMEOUT_S)   # every slice has been written
        except BrokenBarrierError:
            self.pool.check_errors()
            raise RuntimeError("worker died or barrier broken")
        finally:
            stop.set()
            wd.join()
        self.pool.check_errors()
        if self.partition == "slice":
            return self.out                           # no result copying
        out = self._red                               # partition by clients: reduce
        np.copyto(out, self.partials[0])
        for k in range(1, W):
            np.add(out, self.partials[k], out=out)
        return out

    def aggregate(self, U, w):
        """End-to-end: write the updates into shared memory once, then compute.
        WARNING: the returned array is an internal buffer (self.out) that is
        overwritten on the next call. Call .copy() if you need to keep it.
        """
        N, P = U.shape
        if self.block is None or (self.block.N, self.block.P) != (N, P):
            self.use_block(SharedUpdateBlock(N, P), own=True)
        if not np.shares_memory(U, self.block.U):
            np.copyto(self.block.U, U)
            np.copyto(self.block.w, w)
        return self.aggregate_resident()

    def close(self):
        self.release()


class NumbaAggregator:
    """Optional baseline: Numba parallel CPU kernel in one process (Khan et al. [6])."""
    name = "numba"

    def __init__(self, threads):
        import numba
        self.numba = numba
        self.threads = threads
        self._kernel = _get_numba_kernel()
        self.out = None

    def aggregate(self, U, w):
        self.numba.set_num_threads(self.threads)
        P = U.shape[1]
        if self.out is None or self.out.size != P:
            self.out = np.empty(P, DTYPE)
        self._kernel(U, w, self.out)
        return self.out

    def close(self):
        pass


_NUMBA_KERNEL = None


def _get_numba_kernel():
    global _NUMBA_KERNEL
    if _NUMBA_KERNEL is None:
        from numba import njit, prange

        @njit(parallel=True)
        def kernel(U, w, out):
            N, P = U.shape
            T = 4096                                  # tile of parameters per task
            nt = (P + T - 1) // T
            for t in prange(nt):
                a = t * T
                b = min(P, a + T)
                acc = np.zeros(b - a, dtype=np.float32)   # private tile -> vectorizes
                for c in range(N):                    # same client order as Scheme A
                    wc = w[c]
                    row = U[c, a:b]
                    for j in range(b - a):
                        acc[j] += wc * row[j]
                out[a:b] = acc
        _NUMBA_KERNEL = kernel
    return _NUMBA_KERNEL


def numba_available():
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False
