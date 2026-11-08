"""
Figures sized for the IEEE two-column paper (column 3.5 in, page 7.16 in).

    python paper/make_paper_figs.py results/macos_final results/fl_macos_20261016_201349.csv
    python paper/make_paper_figs.py --workers 8 --results results/macos_final
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_common import (
    STYLE, MODEL_NAME, MUTED, MODELS, MCOL, MMARK,
    n_axis, legend_top, plain_log
)
from make_plots import load_bench

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "lines.linewidth": 1.2, "lines.markersize": 3.5, "font.family": "serif"})
OUT = os.path.join(os.path.dirname(__file__), "figures")

# Make a deepcopy of STYLE so we do not mutate the shared dictionary (C9)
PAPER_STYLE = copy.deepcopy(STYLE)
SHORT = {"A": "A: sequential", "B": "B: pickling", "C_slice": "C: shm (slice)",
         "C_clients": "C: shm (clients)", "C_e2e": "C: shm slice + write", "numba": "Numba threads"}
for _k, _v in SHORT.items():
    PAPER_STYLE[_k]["label"] = _v


def line(ax, d, x, y, k, err=True, **kw):
    d = d.sort_values(x)
    st = PAPER_STYLE[k]
    ax.plot(d[x], d[y], color=st["color"], marker=st["marker"], label=st["label"], **kw)
    if err and y == "median_ms":
        ax.fill_between(d[x], d["q1_ms"], d["q3_ms"], color=st["color"], alpha=0.15, lw=0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_time(df, W=8):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.05))
    for ax, m in zip(axes, MODELS):
        d = df[df.model == m]
        for k in ["A", "B", "C_slice", "C_clients", "C_e2e", "numba"]:
            dk = d[(d.k == k) & ((d.W == W) | (k == "A"))]
            line(ax, dk, "N", "median_ms", k)
        n_axis(ax, d.N)
        ax.set_yscale("log")
        ax.set_title(MODEL_NAME[m])
    axes[0].set_ylabel("aggregation time (ms)")
    legend_top(fig, axes, 6)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig_time")


def fig_speedup(df, W=8):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95), sharey=True)
    for ax, m in zip(axes, MODELS):
        d = df[(df.model == m) & (df.W == W)]
        for k in ["C_slice", "C_clients", "C_e2e", "numba", "B"]:
            line(ax, d[d.k == k], "N", "speedup_vs_A", k, err=False)
        ax.axhline(1, color=MUTED, lw=0.8, ls="--")
        n_axis(ax, df[df.model == m].N)
        ax.set_title(MODEL_NAME[m])
    axes[0].set_ylabel("speedup vs. A (W = %d)" % W)
    legend_top(fig, axes, 5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig_speedup")


def fig_copy(df, W=8):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))
    ax = axes[0]
    w = df[df.W == W]
    for m in MODELS:
        d = w[w.model == m]
        b = d[d.k == "B"].set_index("N")
        c = d[d.k == "C_slice"].set_index("N")
        ax.plot(b.block_mb, b.median_s / c.median_s, marker=MMARK[m], color=MCOL[m],
                label=MODEL_NAME[m].split(" ")[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("update data N·P·4 B (MB)")
    ax.set_yticks([10, 20, 40, 70])
    plain_log(ax)
    ax.set_ylabel("time B / time C")
    ax.set_title("(a) pickling vs. shared memory")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
              handlelength=1.5, columnspacing=0.6, handletextpad=0.3)
    ax = axes[1]
    for k in ["A", "B", "C_slice", "numba"]:
        d = df[(df.k == k) & ((df.W == W) | (k == "A"))].sort_values("block_mb")
        ax.plot(d.block_mb, d.block_mb / 1024 / d.median_s, ls="none", marker=PAPER_STYLE[k]["marker"],
                color=PAPER_STYLE[k]["color"], label={"A": "A", "B": "B", "C_slice": "C (slice)",
                                                      "numba": "Numba"}[k], ms=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("update data (MB)")
    ax.set_ylabel("throughput (GB/s)")
    ax.set_title("(b) effective throughput")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    plain_log(ax)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=4, handletextpad=0.1, columnspacing=0.6)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_copy")


def fig_scaling(df):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.3))
    for j, m in enumerate(MODELS):
        N = df[df.model == m].N.max()
        d = df[(df.model == m) & (df.N == N)]
        Ws = sorted(d[d.k != "A"].W.unique())
        for k in ["C_slice", "C_e2e", "numba", "B"]:
            line(axes[0][j], d[d.k == k], "W", "speedup_vs_A", k, err=False)
            line(axes[1][j], d[d.k == k], "W", "proc_cpu_cores", k, err=False)
        axes[0][j].plot(Ws, Ws, color=MUTED, lw=0.8, ls=":", label="ideal linear")
        axes[0][j].set_title("%s (N=%d)" % (MODEL_NAME[m], N))
        for i in (0, 1):
            axes[i][j].set_xticks(Ws)
            axes[i][j].set_xticklabels([str(x) for x in Ws])
            axes[i][j].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axes[1][j].set_xlabel("worker processes / threads W")
    axes[0][0].set_ylabel("speedup vs. A")
    axes[1][0].set_ylabel("busy cores (CPU-s / s)")
    legend_top(fig, axes, 6)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.8)
    save(fig, "fig_scaling")


def fig_partition(df, W=8):
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    w = df[df.W == W]
    for m in MODELS:
        d = w[w.model == m]
        s = d[d.k == "C_slice"].set_index("N")
        c = d[d.k == "C_clients"].set_index("N")
        ax.plot(s.block_mb, c.median_s / s.median_s, marker=MMARK[m], color=MCOL[m],
                label=MODEL_NAME[m])
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("update data N·P·4 B (MB)")
    ax.set_ylabel("time(clients) / time(slice)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_partition")


def fig_fl(path):
    d = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.65))
    lab = {"A": ("A", "-", "A"), "B": ("B", "--", "B"), "C": ("C_slice", ":", "C (slice)")}
    for ax, ds, title in zip(axes, ["cifar10", "femnist"], ["CIFAR-10", "FEMNIST"]):
        for s in ["A", "B", "C"]:
            x = d[(d.dataset == ds) & (d.scheme == s)].sort_values("round")
            k, ls, name = lab[s]
            ax.plot(x["round"], x.test_acc * 100, color=PAPER_STYLE[k]["color"], ls=ls,
                    marker=PAPER_STYLE[k]["marker"], markevery=5, label=name, lw=1.1)
        ax.set_title(title)
        ax.set_xlabel("round")
    axes[0].set_ylabel("test accuracy (%)")
    legend_top(fig, axes, 3)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "fig_fl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_pos", nargs="?", default=None, help="Benchmark results folder")
    ap.add_argument("fl_pos", nargs="?", default=None, help="FL csv file path")
    ap.add_argument("--workers", "-W", type=int, default=8, help="Worker count W for scalability plots (default: 8)")
    ap.add_argument("--results", default="results/macos_final", help="Benchmark results folder (default: results/macos_final)")
    ap.add_argument("--fl", default="results/fl_macos_20261016_201349.csv", help="FL csv path (default: results/fl_macos_20261016_201349.csv)")
    args = ap.parse_args()

    results_dir = args.results_pos if args.results_pos is not None else args.results
    fl_csv = args.fl_pos if args.fl_pos is not None else args.fl
    W = args.workers

    os.makedirs(OUT, exist_ok=True)
    df = load_bench(results_dir)
    fig_time(df, W=W)
    fig_speedup(df, W=W)
    fig_copy(df, W=W)
    fig_scaling(df)
    fig_partition(df, W=W)
    if os.path.exists(fl_csv):
        fig_fl(fl_csv)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
