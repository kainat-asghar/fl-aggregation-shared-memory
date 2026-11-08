"""
Scalability track: benchmark Scheme A / B / C (+ optional Numba) on synthetic
float32 updates at real model sizes (Table IV of the methodology).

    python benchmark.py --quick           # 1-2 minute smoke test
    python benchmark.py                   # full Table IV sweep (~30-60 min)
    python benchmark.py --workers 1 2 4 8 16          # include hyper-threading
    python benchmark.py --models resnet18 --clients 10 25 50 75

Outputs (results/<os>_<timestamp>/):
    summary.csv  one row per (model, N, scheme, W): median time, spread, speedup,
                 efficiency, correctness, CPU utilization, peak memory
    raw.csv      one row per timed repetition
    startup.csv  worker-pool start-up time per W (excluded from round times)
    env.json     machine / OS / start method / library versions
"""
import aggregators as ag   # first: pins BLAS to one thread before numpy loads

import argparse
import csv
import gc
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime

import numpy as np
import psutil

# Parameter counts of the models in Table IV (float32 -> 4 bytes each).
MODELS = {
    "lenet": 62_006,          # LeNet-5, 3x32x32 input, 10 classes
    "mobilenetv2": 3_504_872,  # torchvision mobilenet_v2()
    "resnet18": 11_173_962,   # ResNet-18 with a 10-class head
}
TABLE_IV = {
    "lenet": [10, 50, 100, 250, 500, 1000],
    "mobilenetv2": [10, 50, 100, 250],
    "resnet18": [10, 25, 50, 75],
}
GIB = 2 ** 30
WINDOWS = os.name == "nt"


# ------------------------------------------------------------------ environment
def cpu_name():
    if WINDOWS:
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


def os_label():
    return {"nt": "windows"}.get(os.name, sys.platform.replace("darwin", "macos"))


def env_info(args, start_method):
    vm = psutil.virtual_memory()
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "os": os_label(),
        "platform": platform.platform(),
        "start_method": start_method,
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gib": round(vm.total / GIB, 2),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "kernel": args.kernel,
        "reps": args.reps,
        "block_cap_gib": args.block_cap_gib,
        "pickle_batch_mb": args.pickle_batch_mb,
        "model_params": MODELS,
        "args": vars(args),
    }
    try:
        info["numpy_blas"] = np.show_config(mode="dicts").get(
            "Build Dependencies", {}).get("blas", {}).get("name")
    except Exception:
        pass
    if ag.numba_available():
        import numba
        info["numba"] = numba.__version__
    try:
        bat = psutil.sensors_battery()
        if bat is not None:
            info["on_mains_power"] = bool(bat.power_plugged)
    except Exception:
        pass
    return info


# ------------------------------------------------------------- resource monitor
def _proc_mem(p, parent):
    """Parent: resident set (includes the shared block it touched, counted once).
    Workers: private memory only, so the shared block is not double counted."""
    try:
        if parent:
            return p.memory_info().rss
        if WINDOWS:
            return p.memory_info().private
        try:
            return p.memory_full_info().uss
        except (psutil.AccessDenied, AttributeError):
            return p.memory_info().rss
    except psutil.Error:
        return 0


class ResourceMonitor:
    """Samples memory in a background thread; CPU utilization is taken as the
    system-wide per-core average over the monitored window. Used only in a
    separate instrumented run so it never perturbs the timed repetitions."""

    def __init__(self, worker_pids, interval=0.02):
        self.parent = psutil.Process()
        self.workers = [psutil.Process(p) for p in worker_pids]
        self.interval = interval
        self._stop = threading.Event()

    def _total_mem(self):
        return _proc_mem(self.parent, True) + sum(_proc_mem(p, False) for p in self.workers)

    def _cpu_seconds(self):
        s = 0.0
        for p in [self.parent] + self.workers:
            try:
                t = p.cpu_times()
                s += t.user + t.system
            except psutil.Error:
                pass
        return s

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._total_mem())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.idle = self.peak = self._total_mem()
        self.cpu0 = self._cpu_seconds()
        psutil.cpu_percent(percpu=True)        # reset the system counters
        self.t0 = time.perf_counter()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        percore = psutil.cpu_percent(percpu=True)
        self._stop.set()
        self._th.join()
        self.peak = max(self.peak, self._total_mem())
        self.result = {
            "sys_cpu_mean_pct": round(float(np.mean(percore)), 1),
            "sys_cpu_max_core_pct": round(float(np.max(percore)), 1),
            "cpu_per_core_pct": ";".join("%.0f" % x for x in percore),
            "proc_cpu_cores": round((self._cpu_seconds() - self.cpu0) / self.wall, 2),
            "idle_mem_mb": round(self.idle / 2 ** 20, 1),
            "peak_mem_mb": round(self.peak / 2 ** 20, 1),
            "extra_mem_mb": round((self.peak - self.idle) / 2 ** 20, 1),
        }


# ---------------------------------------------------------------- synthetic data
def fedscale_weights(N, rng):
    """Heavy-tailed client dataset sizes (FedScale-like), normalized weights."""
    n = np.maximum(10, rng.lognormal(mean=np.log(300), sigma=1.0, size=N)).astype(np.int64)
    w = (n / n.sum()).astype(np.float32)
    return w, n


def fill_updates(U, rng, rows_per_chunk=8):
    """Random float32 deltas, generated directly into the shared block."""
    for i in range(0, U.shape[0], rows_per_chunk):
        chunk = U[i:i + rows_per_chunk]
        rng.standard_normal(dtype=np.float32, out=chunk)
        chunk *= np.float32(0.01)


def straggler_arrivals(n_samples, rng):
    """Virtual-clock arrival time of each client update (compute ~ data size x
    device speed, plus network delay). Recorded, never slept on, and excluded
    from the measured aggregation time."""
    speed = rng.lognormal(0.0, 0.5, size=n_samples.size)
    compute = n_samples * 1e-3 * speed
    network = rng.lognormal(np.log(0.2), 0.6, size=n_samples.size)
    return compute + network


# ------------------------------------------------------------------- benchmark
def quantiles(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(med), float(q1), float(q3)


def build_variants(args, pools, block, aggA, aggs, ring):
    """Each variant: (scheme, partition, W, fn, worker_pids)."""
    U, w = block.U, block.w
    v = []
    if "A" in args.schemes:
        v.append(("A_sequential", "-", 1, lambda: aggA.aggregate(U, w), []))
    for W in args.workers:
        pool = pools[W]
        if "B" in args.schemes:
            b = aggs[("B", W)]
            v.append(("B_pickle", "clients", W, (lambda b=b: b.aggregate(U, w)), pool.pids))
        if "C" in args.schemes:
            c = aggs[("C_slice", W)]
            c.use_block(block)
            v.append(("C_shm", "slice", W, c.aggregate_resident, pool.pids))
        if "C_clients" in args.schemes:
            c = aggs[("C_clients", W)]
            c.use_block(block)
            v.append(("C_shm", "clients", W, c.aggregate_resident, pool.pids))
        if "numba" in args.schemes and W in aggs.get("numba_ok", ()):
            nb = aggs[("numba", W)]
            v.append(("numba", "tiles", W, (lambda nb=nb: nb.aggregate(U, w)), []))
    e2e = []
    if "C_e2e" in args.schemes:
        src_w = w.copy()
        for W in args.workers:
            c = aggs[("C_slice", W)]

            def fn(c=c):
                # one-time write of the N received updates into shared memory (vectorized take)
                K = ring.shape[0]
                idx = np.arange(U.shape[0]) % K
                np.take(ring, idx, axis=0, out=U, mode="clip")
                np.copyto(w, src_w)
                return c.aggregate_resident()
            e2e.append(("C_shm_e2e", "slice", W, fn, pools[W].pids))
    return v, e2e


def _measure(args, model, P, N, block, ring, arrivals, pools, aggA, aggs, env, writers):
    block_bytes = block.nbytes
    variants, e2e = build_variants(args, pools, block, aggA, aggs, ring)
    common = dict(os=env["os"], start_method=env["start_method"], kernel=args.kernel,
                  model=model, P=P, N=N, block_mb=round(block_bytes / 2 ** 20, 1))
    stats = {}

    # ---- warm-up (discarded) + correctness check against Scheme A
    ref = aggA.aggregate(block.U, block.w).copy()
    for key in variants:
        name, part, W, fn, _ = key
        tw = time.perf_counter()
        out = fn()
        tw = time.perf_counter() - tw
        diff = float(np.max(np.abs(out - ref)))
        ok = bool(np.allclose(ref, out, atol=1e-6))
        stats[key[:3]] = dict(warmup_s=tw, maxdiff=diff, allclose=ok, times=[])
        if not ok:
            print("  !! %s W=%d does NOT match Scheme A (max diff %.3g)" % (name, W, diff))
    if e2e:
        e2e[0][3]()                                # first write -> data is now ring rows
        ref2 = aggA.aggregate(block.U, block.w).copy()
        for key in e2e:
            name, part, W, fn, _ = key
            tw = time.perf_counter()
            out = fn()
            tw = time.perf_counter() - tw
            diff = float(np.max(np.abs(out - ref2)))
            stats[key[:3]] = dict(warmup_s=tw, maxdiff=diff,
                              allclose=bool(np.allclose(ref2, out, atol=1e-6)), times=[])
    allv = variants + e2e
    del ref

    # ---- timed repetitions, interleaved in a shuffled order to spread drift
    order_rng = np.random.default_rng(args.seed + N)
    for r in range(args.reps):
        for i in order_rng.permutation(len(allv)):
            key = allv[i]
            gc.disable()
            t = time.perf_counter()
            key[3]()
            t = time.perf_counter() - t
            gc.enable()
            stats[key[:3]]["times"].append(t)
            writers["raw"].writerow(dict(common, scheme=key[0], partition=key[1],
                                         W=key[2], rep=r, time_s="%.6f" % t))

    # ---- instrumented run: CPU utilization + peak memory (not used for timing)
    for key in allv:
        with ResourceMonitor(key[4]) as mon:
            it = 0
            while it == 0 or time.perf_counter() - mon.t0 < args.monitor_s:
                key[3]()
                it += 1
        stats[key[:3]].update(mon.result)

    # ---- summary
    a_key = next((k for k in allv if k[0] == "A_sequential"), None)
    a_med = quantiles(stats[a_key[:3]]["times"])[0] if a_key else float("nan")
    for key in allv:
        s = stats[key[:3]]
        med, q1, q3 = quantiles(s["times"])
        sp = a_med / med
        row = dict(common, scheme=key[0], partition=key[1], W=key[2], reps=len(s["times"]),
                   median_s="%.6f" % med, q1_s="%.6f" % q1, q3_s="%.6f" % q3,
                   min_s="%.6f" % min(s["times"]), max_s="%.6f" % max(s["times"]),
                   warmup_s="%.6f" % s["warmup_s"],
                   speedup_vs_A="%.3f" % sp, efficiency="%.3f" % (sp / key[2]),
                   maxdiff_vs_A="%.3g" % s["maxdiff"], allclose=s["allclose"],
                   sim_arrival_max_s="%.3f" % arrivals.max(),
                   **{k: s[k] for k in ("sys_cpu_mean_pct", "sys_cpu_max_core_pct", "proc_cpu_cores",
                                        "idle_mem_mb", "peak_mem_mb", "extra_mem_mb",
                                        "cpu_per_core_pct")})
        writers["summary"].writerow(row)
        print("  %-13s %-8s W=%-2d  median %9.2f ms  speedup %5.2fx  peak %7.0f MB  %s"
              % (key[0], key[1], key[2], med * 1e3, sp, s["peak_mem_mb"],
                 "ok" if s["allclose"] else "MISMATCH"))
    for f in writers["files"]:
        f.flush()


def run_config(args, model, P, N, pools, aggA, aggs, env, writers, rng):
    block_bytes = (N * P + N) * 4
    tag = "%s N=%d (%.2f GiB)" % (model, N, block_bytes / GIB)
    if block_bytes > args.block_cap_gib * GIB:
        print("  skip %s: above the %.1f GiB block cap" % (tag, args.block_cap_gib))
        return
    need = block_bytes * 1.1 + 1.5 * GIB
    avail = psutil.virtual_memory().available
    if avail < need:
        print("  skip %s: needs ~%.1f GiB free RAM, %.1f GiB available"
              % (tag, need / GIB, avail / GIB))
        return

    print("\n=== %s ===" % tag)
    t0 = time.perf_counter()
    block = ag.SharedUpdateBlock(N, P)
    fill_updates(block.U, rng)
    w, n_samples = fedscale_weights(N, rng)
    block.w[:] = w
    arrivals = straggler_arrivals(n_samples, rng)
    ring = block.U[:min(N, 4)].copy()            # source rows for the e2e write test
    print("  generated updates in %.1fs" % (time.perf_counter() - t0))

    try:
        # all views/closures live in _measure's frame, so they are gone
        # before the shared blocks are closed (required on Windows)
        _measure(args, model, P, N, block, ring, arrivals, pools, aggA, aggs, env, writers)
    finally:
        # release shared blocks (workers detach first)
        for a in aggs.values():
            if isinstance(a, ag.SharedMemoryAggregator):
                a.release()
        block.close()
        gc.collect()


SUMMARY_FIELDS = ["os", "start_method", "kernel", "model", "P", "N", "block_mb", "scheme",
                  "partition", "W", "reps", "median_s", "q1_s", "q3_s", "min_s", "max_s",
                  "warmup_s", "speedup_vs_A", "efficiency", "maxdiff_vs_A", "allclose",
                  "sys_cpu_mean_pct", "sys_cpu_max_core_pct", "proc_cpu_cores", "idle_mem_mb",
                  "peak_mem_mb", "extra_mem_mb", "sim_arrival_max_s", "cpu_per_core_pct"]
RAW_FIELDS = ["os", "start_method", "kernel", "model", "P", "N", "block_mb", "scheme",
              "partition", "W", "rep", "time_s"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--clients", nargs="+", type=int,
                   help="override client counts for every model (default: Table IV)")
    p.add_argument("--workers", nargs="+", type=int, default=None,
                   help="worker counts W (default: 1 2 4 8, capped at physical cores)")
    p.add_argument("--schemes", nargs="+",
                   default=["A", "B", "C", "C_clients", "C_e2e", "numba"],
                   choices=["A", "B", "C", "C_clients", "C_e2e", "numba"],
                   help="C = shared memory, slice partition (compute-only); "
                        "C_clients = client partition; C_e2e = C incl. the write into shm")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--kernel", choices=["blocked", "axpy", "gemv"], default="blocked",
                   help="inner NumPy kernel, identical in every scheme (see aggregators.accumulate)")
    p.add_argument("--block-cap-gib", type=float, default=3.5)
    p.add_argument("--pickle-batch-mb", type=float, default=64)
    p.add_argument("--monitor-s", type=float, default=1.0,
                   help="minimum length of the instrumented CPU/memory run")
    p.add_argument("--start-method", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--high-priority", action="store_true",
                   help="raise parent + worker priority to reduce scheduling noise")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true", help="small smoke test")
    a = p.parse_args()
    phys = psutil.cpu_count(logical=False) or 4
    if a.quick:
        a.models = [m for m in a.models if m != "resnet18"] or ["lenet"]
        a.clients = a.clients or [10, 50]
        a.workers = a.workers or [1, 2, 4]
        a.reps = min(a.reps, 3)
        a.monitor_s = 0.3
    if a.workers is None:
        a.workers = [x for x in (1, 2, 4, 8) if x <= phys] or [1]
    return a


def set_high_priority(pids):
    for pid in [os.getpid()] + list(pids):
        try:
            p = psutil.Process(pid)
            p.nice(psutil.HIGH_PRIORITY_CLASS if WINDOWS else -5)
        except (psutil.Error, AttributeError):
            pass


def main():
    args = parse_args()
    start_method = args.start_method or ag.default_start_method()
    env = env_info(args, start_method)
    outdir = os.path.join(args.out, "%s_%s" % (env["os"], datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(outdir, exist_ok=True)
    print("CPU: %s | %s physical / %s logical cores | RAM %.1f GiB"
          % (env["cpu"], env["physical_cores"], env["logical_cores"], env["ram_gib"]))
    print("OS: %s | start method: %s | kernel: %s | results -> %s"
          % (env["platform"], start_method, args.kernel, outdir))
    if env.get("on_mains_power") is False:
        print("WARNING: running on battery - plug in and use the best-performance power mode.")

    # ---- persistent worker pools (one per W), start-up timed separately
    pools, aggs = {}, {}
    with open(os.path.join(outdir, "startup.csv"), "w", newline="") as f:
        sw = csv.DictWriter(f, fieldnames=["os", "start_method", "W", "startup_s"])
        sw.writeheader()
        for W in args.workers:
            pools[W] = ag.WorkerPool(W, start_method, kernel=args.kernel)
            print("  pool W=%-2d started in %.3fs" % (W, pools[W].startup_s))
            sw.writerow(dict(os=env["os"], start_method=start_method, W=W,
                             startup_s="%.4f" % pools[W].startup_s))
    env["pool_startup_s"] = {W: pools[W].startup_s for W in args.workers}
    if args.high_priority:
        set_high_priority([pid for pl in pools.values() for pid in pl.pids])

    aggA = ag.SequentialAggregator(args.kernel)
    for W in args.workers:
        aggs[("B", W)] = ag.PickleAggregator(pools[W], args.pickle_batch_mb)
        aggs[("C_slice", W)] = ag.SharedMemoryAggregator(pools[W], "slice")
        aggs[("C_clients", W)] = ag.SharedMemoryAggregator(pools[W], "clients")
    if "numba" in args.schemes:
        if ag.numba_available():
            import numba
            ok = [W for W in args.workers if W <= numba.config.NUMBA_NUM_THREADS]
            aggs["numba_ok"] = ok
            for W in ok:
                aggs[("numba", W)] = ag.NumbaAggregator(W)
            print("  compiling Numba kernel ...")
            aggs[("numba", ok[0])].aggregate(np.ones((2, 64), np.float32),
                                             np.ones(2, np.float32))
        else:
            print("  numba not installed - skipping the optional Numba baseline")

    with open(os.path.join(outdir, "env.json"), "w") as f:
        json.dump(env, f, indent=2, default=str)

    rng = np.random.default_rng(args.seed)
    fs = open(os.path.join(outdir, "summary.csv"), "w", newline="")
    fr = open(os.path.join(outdir, "raw.csv"), "w", newline="")
    writers = {"summary": csv.DictWriter(fs, SUMMARY_FIELDS),
               "raw": csv.DictWriter(fr, RAW_FIELDS), "files": (fs, fr)}
    writers["summary"].writeheader()
    writers["raw"].writeheader()
    t_all = time.perf_counter()
    try:
        for model in args.models:
            for N in (args.clients or TABLE_IV[model]):
                run_config(args, model, MODELS[model], N, pools, aggA, aggs, env, writers, rng)
    except KeyboardInterrupt:
        print("\ninterrupted - partial results kept")
    finally:
        fs.close()
        fr.close()
        for a in aggs.values():
            if isinstance(a, ag.SharedMemoryAggregator):
                a.close()
        for pl in pools.values():
            pl.close()
    print("\nDone in %.1f min. Results in %s" % ((time.perf_counter() - t_all) / 60, outdir))
    print("Next: python make_plots.py %s" % outdir)


if __name__ == "__main__":
    main()
