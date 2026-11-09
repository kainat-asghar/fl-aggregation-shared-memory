# FL aggregation experiments: sequential vs pickling vs shared memory

This repository evaluates federated learning model aggregation mechanisms on commodity multi-core CPUs.
All experimental results reported in the paper were collected in a single run on
one machine: an AMD Ryzen 7 7735HS laptop (8 physical / 16 logical cores, 16 GB
DDR5, Windows 11, Python 3.12.10, `spawn` start method). Everything executes
entirely on CPU; no GPU is used.

Earlier partial runs from two other machines (an Apple M4 and an Intel
i5-10500H under Linux) are kept for reference in
`results/_archive_other_machines/` and are **not** used by the paper. Timings
from different machines must never be combined in one table or plot.

| File | What it does | Research question |
|---|---|---|
| `aggregators.py` | Scheme A (sequential), B (multiprocessing + pickling), C (multiprocessing + `shared_memory`, slice or client partitioning), optional Numba baseline. All four share the interface `aggregate(updates, weights) -> global_model`. | all |
| `benchmark.py` | Scalability track: synthetic float32 updates at real model sizes (Table II), swept over client count N, model size P and worker count W. | RQ1, RQ2, RQ3 |
| `fl_train.py` | Correctness track: real FedAvg training on CIFAR-10 (Dirichlet α = 0.5) and FEMNIST, run once per scheme. Both datasets come from the Hugging Face hub (`uoft-cs/cifar10`, `flwrlabs/femnist`). If that fails, CIFAR-10 falls back to torchvision. | correctness |
| `bench_stream.py` | STREAM-style memory-bandwidth microbenchmark. Measures the sustained DRAM ceiling the roofline argument rests on, including an `fma` kernel identical to the aggregation inner loop. | RQ3 |
| `make_plots.py` | Builds benchmark figures (PNG + PDF) and `tables.md` from the CSV files. | – |
| `plot_common.py` | Shared styles, color maps, markers, and helpers for plotting scripts. | – |
| `tests/test_aggregators.py` | Unit tests for aggregation kernels, boundary partitioning edge cases, and scheme equivalence. | correctness |

## 1. Setup

Python 3.9 to 3.14 are supported.

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Alternatively, for exact pinned versions:
pip install -r requirements-lock.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

If `numba` fails to install on your Python version (e.g., Python 3.14), delete or skip it from `requirements.txt`. It is an optional baseline, and the benchmark skips it automatically.

## 2. Before every measured run

* Plug in the charger and set your power profile to **Best performance**.
* Close browsers, background synchronizations, and resource-heavy applications.
* Optional (Windows): add the project folder to Windows Defender exclusions to prevent process startup overhead.

## 3. Run the experiments

```bash
# (a) smoke test, about 1 minute. Check that every line ends with "ok"
python benchmark.py --quick

# (b) run test suite
pytest tests/ -v

# (c) full scalability sweep (Table II: LeNet up to 1000 clients, MobileNetV2 up to 250, ResNet-18 up to 75)
#     W = 1, 2, 4, 8.
python benchmark.py

# (d) correctness track: 20 clients, 20 rounds, CIFAR-10 + FEMNIST, schemes A, B, C
python fl_train.py --quick        # 2-minute check first
python fl_train.py

# (e) memory-bandwidth ceiling for the roofline argument (RQ3), ~2 min
#     run this on an otherwise idle machine
python bench_stream.py

# (f) figures + tables (use the folder / file names printed by the runs)
python make_plots.py results/<run_folder> --fl results/<fl_run>.csv
python paper/make_paper_figs.py results/<run_folder>
```

Press Ctrl+C at any time. Every finished configuration is already saved.

Useful options: `--models`, `--clients`, `--workers`, `--schemes`, `--reps`,
`--kernel {blocked,axpy,gemv}`, and `--block-cap-gib` (lower it if Windows
starts paging). Run `python benchmark.py -h` for the full list.

## 4. Which results the paper uses

| Folder / file | Role |
|---|---|
| `results/windows_20261109_202153/` | **Canonical scalability sweep.** Every number in Tables V and IX and Figs. 3, 4, 6, 7 comes from here. 294/294 configurations pass `allclose`; max abs diff 5.59e-9. |
| `results/fl_windows_20261109_004227.csv` (+ `_summary.json`) | **Correctness track.** Table IV and Fig. 2. Scheme C is bit-identical to A in every round of both datasets. |
| `results/stream_windows_20261109_004931.{csv,json}` | **Memory-bandwidth ceiling.** Table VI and the roofline argument in Section V-D. |
| `results/tile_sweep/` | **Tile-size sweep.** Fig. 5: tile size (16 KB - 2 MB) crossed with W, on MobileNetV2 N=50 and ResNet-18 N=25. The data the tile rule was derived from. |
| `results/ablation_fixed_1mb_tile/` | **Ablation arm**, Table VII: the full sweep with the historical fixed 1 MB tile. |
| `results/ablation_llc_rule_tile/` | **Ablation arm**, Table VII: the full sweep with an LLC-budgeted tile, the intermediate rule that fixes the collapse but still forfeits ~25%. |
| `results/_archive_other_machines/` | Superseded runs from an Apple M4 and an Intel i5-10500H. **Not used by the paper.** Kept only so the earlier work is not lost. |

Do not mix rows from different folders: the machines differ, and so do the tile
rules. The three full sweeps differ **only** in the tile rule, so they are
directly comparable to each other; everything else about them is identical.

## 5. Outputs

`results\windows_<timestamp>\`

| File | Contents |
|---|---|
| `summary.csv` | One row per (model, N, scheme, W) with: median, IQR, min and max time; speedup vs A; efficiency = speedup / W; max abs diff vs A and the `allclose` result; CPU utilization (system mean, busiest core, per core, busy cores of the process tree); idle, peak and extra memory |
| `raw.csv` | Every timed repetition, for box plots or statistical tests |
| `startup.csv` | Worker-pool start-up time per W under `spawn`. It is reported separately and never counted inside a round |
| `env.json` | CPU name, cores, RAM, OS, start method, Python/NumPy versions, battery state, exact parameter counts |
| `figures\` | `fig_rq1_time_vs_clients`, `fig_rq1_speedup_vs_clients`, `fig_rq2_copy_cost`, `fig_rq3_scaling_workers`, `fig_rq3_cpu_memory`, `fig_partitioning`, `fig_pool_startup`, `fig_fl_accuracy` |
| `tables.md` | Ready-to-copy result tables for the paper |

`fl_train.py` writes `results\fl_windows_<timestamp>.csv`, with per-round accuracy,
loss, aggregation time and max parameter difference vs a sequential reference.
It also writes `_summary.json`, with final accuracy per scheme and the maximum
accuracy gap vs Scheme A.

## 6. Implementation details (for the "Experimentation / Implementation" section)

**Common interface and kernel.** Every scheme computes `global = Σ w[c]·u[c]`
with normalized weights `w[c] = n[c]/Σn` (float32). All schemes share the same
inner NumPy kernel (`accumulate`), so measured differences come from the
mechanism and not from different arithmetic. The default `blocked` kernel walks
the parameter vector in tiles. Inside each tile it loops over clients in
order. As a result:

* each update byte is read from memory once;
* the sequential baseline is as cache-friendly as a parallel slice, so speedups
  are not a cache artifact;
* clients are summed in the same order per element, so Scheme C with slice
  partitioning is bit-identical to Scheme A.

**Tile size is derived from the worker count.** Each worker keeps two
tile-sized buffers hot (its slice of the accumulator, plus scratch). The
governing cache is the **per-core L2**, not the shared LLC. `tile_elems(W, m)`
gives a lone worker a full L2-sized tile and contending workers `L2/(2W)`,
clamped to `[L2/8, L2/4]`, never exceeding the slice length, so a model smaller
than one tile runs as a single untiled pass. With a 512 KB L2 this gives
512 KB at W=1, 128 KB at W=2 and 64 KB at W=4 and 8 — which is exactly the
measured optimum at each W (`results/tile_sweep/`).

This matters a great deal. The historical fixed 1 MB tile inverts the scaling
curve: at W=8 it makes shared memory *slower than sequential* (0.51x on
ResNet-18) where the tuned rule reaches 1.65x. An LLC-budgeted rule fixes the
inversion but still forfeits ~25%. Both arms are kept as full sweeps
(`results/ablation_fixed_1mb_tile/`, `results/ablation_llc_rule_tile/`) for
Table VII.

Two bounds set the optimum, and both are visible in Fig. 5 of the paper: above
it the workers' resident sets contend (a 2 MB tile at W=8 is 4x slower than
64 KB), below it per-tile NumPy call overhead dominates (a 16 KB tile nearly
doubles even single-threaded time). The divisors were tuned on this machine;
the shape of the rule is what transfers, so re-run the sweep on new hardware.

The tile changes only the traversal order, never the arithmetic, so results are
bitwise independent of it. Set `FL_AGG_TILE_KB` to override the rule and
reproduce either ablation, e.g. `FL_AGG_TILE_KB=1024` for the old fixed
behaviour.

`--kernel axpy` runs the literal per-client loop of the pseudocode, and
`--kernel gemv` runs one single-threaded BLAS call. BLAS is pinned to one
thread in every process, so Scheme A really uses one core.

**Scheme B (pickling).** A persistent pool of W processes is created once.
Clients are split into W contiguous chunks. The parent explicitly serializes
each chunk with `pickle.dumps` and sends it through a `multiprocessing.Pipe`,
in batches of at most 64 MB (`--pickle-batch-mb`). Batches go round-robin so
workers compute while the parent pickles the next batch. Batching keeps memory
bounded on a 16 GB machine, and every byte of the N·P updates is still copied
exactly once. Each worker returns its pickled partial vector, and the parent
sums the W partials.

**Scheme C (shared memory).** One `multiprocessing.shared_memory` block holds
the N×P update matrix followed by the N weights. A second block holds the
P-value output. Workers receive only metadata: block names, N, P, partition
mode and worker index. They attach once (the handle is cached across rounds)
and build zero-copy NumPy views.

* **Slice partitioning (default):** worker k averages a contiguous,
  64-byte-aligned slice of P over all N clients and writes it straight into the
  output. No locks and no reduction are needed.
* **Client partitioning:** each worker writes a full partial vector into a
  shared W×P buffer, and the parent reduces it.

A `multiprocessing.Barrier(W+1)` marks the end of each round. Two timings are
reported:

* compute-only (`C_shm`): the updates are already resident in shared memory;
* end-to-end (`C_shm_e2e`): includes the parent writing all N updates into the
  block.

The parent releases the blocks after each configuration, after the workers
detach. On Windows this matters, because a block is freed only when its last
handle is closed.

**Process start-up.** Windows uses `spawn`. Pools are started before any data
exists, timed separately (`startup.csv`), and reused for every round. The first
round of each configuration, which includes the one-time shared-memory attach,
is a discarded warm-up.

**Protocol per configuration.**

1. Generate the updates directly into the shared block, so no duplicate copy
   exists. Schemes A and B read the same memory.
2. Run one warm-up per scheme and check correctness:
   `allclose(global_A, global_X, atol=1e-6)` and max abs diff.
3. Run 5 timed repetitions, interleaved across schemes in a shuffled order to
   spread thermal drift, with GC disabled during each timed call.
4. Run a separate instrumented pass of at least 1 s per scheme for CPU and
   memory, so the sampler never perturbs the timings.

Medians and IQR are reported.

**Memory accounting.** Peak memory is the parent's working set plus the
**private** bytes of each worker. The shared block is therefore counted once,
through the parent. It is sampled every 20 ms.

**FedScale-style realism.**

* Client dataset sizes are log-normal (heavy-tailed) and set the weights.
* Straggler delays use a virtual clock: arrival = data size × log-normal device
  speed + log-normal network delay.
* In `fl_train.py`, updates are aggregated in arrival order.
* Delays are recorded (`sim_round_s`, `sim_arrival_max_s`) but never slept on,
  so they are excluded from aggregation time, as the methodology states.

**Memory cap.** Raw update blocks are capped at 3.5 GiB, which gives the Table IV
client counts. A configuration is skipped if free RAM is below
block + 1.5 GiB.

### Differences from the methodology text (please update the paper)

1. **No Flower.** `fl_train.py` uses a small FedAvg loop written directly in
   PyTorch. Flower's simulation engine depends on Ray, which is unreliable on
   Windows, and calling our own aggregators directly keeps the comparison
   controlled. Suggested wording: *"a lightweight FedAvg simulation in PyTorch
   with 20 clients"*.
2. **Exact parameter counts.**
   * LeNet: 62,006 (CIFAR-10) and ≈66K (FEMNIST, 62 classes).
   * MobileNetV2: 3,504,872, i.e. **3.5M**, not 3.4M.
   * ResNet-18 with a 10-class head: 11,173,962.
3. **Scheme B batching.** Scheme B streams each worker's chunk in ≤ 64 MB pickled
   batches instead of one message. Total bytes copied are unchanged; only the
   peak memory is bounded.
4. **Kernel.** The default kernel is the cache-blocked version of the pseudocode
   loop, explained above.

## 7. Troubleshooting

* **"skip … needs ~X GiB free RAM"**: close other programs, or pass `--block-cap-gib 2.5`.
* **`RuntimeError … freeze_support` / workers crash at start**: always run the
  scripts directly (`python benchmark.py`), not from an interactive console or notebook.
* **Very slow pool start-up**: this usually comes from antivirus scanning. See §2.
* **A download fails**: run one dataset at a time (`--datasets cifar10` or `--datasets femnist`) and retry.
  The processed data is cached in `data\cifar10.npz` and `data\femnist_*.npz`, so each download happens only once.
