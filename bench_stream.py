"""
STREAM-style memory-bandwidth microbenchmark (McCalpin).

The aggregation kernel has an arithmetic intensity of about 1/4 FLOP per byte,
so its ceiling is set by sustained DRAM bandwidth rather than by peak FLOPs.
This script measures that ceiling on the current machine, which is the number
the roofline argument in the paper rests on.

    python bench_stream.py                      # 1..logical-cores threads
    python bench_stream.py --threads 1 2 4 8
    python bench_stream.py --array-mb 512 --reps 10

Kernels (per STREAM, plus one that matches our read pattern):

    copy    a[:] = b[:]              2 bytes moved per element-byte  (1R + 1W)
    scale   a[:] = q * b[:]          1R + 1W
    add     a[:] = b[:] + c[:]       2R + 1W
    triad   a[:] = b[:] + q * c[:]   2R + 1W
    read    s   = sum over tiles     1R
    fma     acc[tile] += q * b[:]    1R          <- the FedAvg aggregation kernel

The `fma` kernel is the one that matters for the roofline: it is exactly the
inner loop of aggregators.accumulate(), a pure streaming read of the update
block into a cache-resident accumulator tile. A plain reduction (`read`) is
limited by the dependency chain of the summation at low thread counts and so
understates what the memory system can actually deliver; `fma` does not.

Bandwidth counts every byte that must cross the memory bus, so a write-allocate
read is not counted; the reported numbers are therefore a lower bound on true
bus traffic, exactly as in the reference STREAM implementation.

Outputs results/stream_<os>_<timestamp>.csv and .json.
"""
import argparse
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime

# single-threaded BLAS: the parallelism here must come from our own threads
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import psutil

DTYPE = np.float32
ITEM = np.dtype(DTYPE).itemsize
TILE = 262_144          # 1 MB of float32, same tile as aggregators.accumulate


def os_label():
    return {"nt": "windows"}.get(os.name, sys.platform.replace("darwin", "macos"))


def cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    if sys.platform == "darwin":
        try:
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


# --------------------------------------------------------------------- kernels
# Each kernel works on a contiguous slice [lo, hi) and is run inside one thread.
# NumPy releases the GIL for these element-wise loops, so threads really do run
# in parallel; this mirrors the Numba baseline rather than Scheme C, but the
# sustained bandwidth ceiling it measures is a property of the machine.

def k_copy(a, b, c, q, lo, hi):
    for s in range(lo, hi, TILE):
        e = min(hi, s + TILE)
        a[s:e] = b[s:e]


def k_scale(a, b, c, q, lo, hi):
    for s in range(lo, hi, TILE):
        e = min(hi, s + TILE)
        np.multiply(b[s:e], q, out=a[s:e])


def k_add(a, b, c, q, lo, hi):
    for s in range(lo, hi, TILE):
        e = min(hi, s + TILE)
        np.add(b[s:e], c[s:e], out=a[s:e])


def k_triad(a, b, c, q, lo, hi):
    tmp = np.empty(min(TILE, hi - lo), DTYPE)
    for s in range(lo, hi, TILE):
        e = min(hi, s + TILE)
        t = tmp[:e - s]
        np.multiply(c[s:e], q, out=t)
        np.add(b[s:e], t, out=a[s:e])


def k_read(a, b, c, q, lo, hi):
    acc = 0.0
    for s in range(lo, hi, TILE):
        e = min(hi, s + TILE)
        acc += float(b[s:e].sum(dtype=np.float32))
    a[lo] = acc


def k_fma(a, b, c, q, lo, hi):
    """Exactly aggregators.accumulate(): stream b into a cache-resident tile."""
    acc = np.zeros(TILE, DTYPE)
    tmp = np.empty(TILE, DTYPE)
    for s in range(lo, hi, TILE):
        e = min(hi, s + TILE)
        o, t = acc[:e - s], tmp[:e - s]
        np.multiply(b[s:e], q, out=t)
        np.add(o, t, out=o)
    a[lo] = acc[0]


# bytes crossing the bus per array element, per kernel
BYTES_PER_ELEM = {"copy": 2 * ITEM, "scale": 2 * ITEM, "add": 3 * ITEM,
                  "triad": 3 * ITEM, "read": 1 * ITEM, "fma": 1 * ITEM}
KERNELS = {"copy": k_copy, "scale": k_scale, "add": k_add,
           "triad": k_triad, "read": k_read, "fma": k_fma}


def run_threaded(fn, a, b, c, q, n, nthreads):
    """Split [0, n) across nthreads and time one full pass."""
    step = -(-n // nthreads)
    bounds = [(k * step, min(n, (k + 1) * step)) for k in range(nthreads)]
    bounds = [(lo, hi) for lo, hi in bounds if hi > lo]
    threads = [threading.Thread(target=fn, args=(a, b, c, q, lo, hi))
               for lo, hi in bounds]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--array-mb", type=float, default=512,
                   help="size of EACH of the three arrays in MB (default 512, "
                        "must be several times the last-level cache)")
    p.add_argument("--reps", type=int, default=10, help="timed passes per kernel")
    p.add_argument("--threads", nargs="+", type=int, default=None,
                   help="thread counts (default: 1 2 4 ... up to logical cores)")
    p.add_argument("--kernels", nargs="+", default=list(KERNELS),
                   choices=list(KERNELS))
    p.add_argument("--tile-kb", type=float, default=1024, help="inner tile size per thread in KB")
    p.add_argument("--out", default="results")
    a_ = p.parse_args()

    logical = psutil.cpu_count(logical=True) or 4
    physical = psutil.cpu_count(logical=False) or logical
    if a_.threads is None:
        th, t = [], 1
        while t <= logical:
            th.append(t)
            t *= 2
        if logical not in th:
            th.append(logical)
        a_.threads = th

    global TILE
    TILE = max(4096, int(a_.tile_kb * 1024 // ITEM))
    n = int(a_.array_mb * 1e6 // ITEM)
    need_gib = 3 * n * ITEM / 2 ** 30
    free_gib = psutil.virtual_memory().available / 2 ** 30
    if need_gib + 0.5 > free_gib:
        sys.exit("need ~%.1f GiB for the three arrays, only %.1f GiB free; "
                 "lower --array-mb" % (need_gib, free_gib))

    print("CPU: %s | %d physical / %d logical cores" % (cpu_name(), physical, logical))
    print("arrays: 3 x %.0f MB float32 (%.2f GiB total), %d reps\n"
          % (a_.array_mb, need_gib, a_.reps))

    rng = np.random.default_rng(0)
    a = np.zeros(n, DTYPE)
    b = (rng.random(n, dtype=np.float32) + 1.0).astype(DTYPE)
    c = (rng.random(n, dtype=np.float32) + 1.0).astype(DTYPE)
    q = DTYPE(3.0)

    rows = []
    for kname in a_.kernels:
        fn = KERNELS[kname]
        moved = n * BYTES_PER_ELEM[kname]
        print("%-6s" % kname, end="", flush=True)
        for nt in a_.threads:
            run_threaded(fn, a, b, c, q, n, nt)          # warm-up, discarded
            times = [run_threaded(fn, a, b, c, q, n, nt) for _ in range(a_.reps)]
            best, med = min(times), float(np.median(times))
            rows.append(dict(kernel=kname, threads=nt,
                             bytes_moved=moved,
                             best_s=round(best, 6), median_s=round(med, 6),
                             best_gb_s=round(moved / best / 1e9, 2),
                             median_gb_s=round(moved / med / 1e9, 2)))
            print("  T=%-2d %6.1f GB/s" % (nt, moved / best / 1e9), end="", flush=True)
        print()

    peak = max(r["best_gb_s"] for r in rows)
    peak_row = next(r for r in rows if r["best_gb_s"] == peak)
    print("\nsustained peak: %.1f GB/s  (%s, %d threads)"
          % (peak, peak_row["kernel"], peak_row["threads"]))
    print("roofline for aggregation at 1/4 FLOP/byte: %.1f GFLOP/s" % (peak / 4))

    os.makedirs(a_.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(a_.out, "stream_%s_%s" % (os_label(), stamp))
    import csv
    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(base + ".json", "w") as f:
        json.dump(dict(timestamp=stamp, os=os_label(), platform=platform.platform(),
                       cpu=cpu_name(), physical_cores=physical, logical_cores=logical,
                       ram_gib=round(psutil.virtual_memory().total / 2 ** 30, 2),
                       python=platform.python_version(), numpy=np.__version__,
                       array_mb=a_.array_mb, reps=a_.reps,
                       peak_gb_s=peak, peak_kernel=peak_row["kernel"],
                       peak_threads=peak_row["threads"], rows=rows), f, indent=2)
    print("wrote %s.csv / .json" % base)


if __name__ == "__main__":
    main()
