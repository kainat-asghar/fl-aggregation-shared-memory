"""
Turn benchmark / FL results into paper figures (PNG + PDF) and markdown tables.

    python make_plots.py results/windows_20261016_120000            # benchmark folder
    python make_plots.py results/windows_... --fl results/fl_windows_....csv

Figures go to <benchmark folder>/figures/, tables to <benchmark folder>/tables.md.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from plot_common import (
    STYLE, MODEL_NAME, INK, MUTED, GRID, scheme_key, n_axis, legend_below
)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "lines.markersize": 5, "figure.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load_bench(folder):
    df = pd.read_csv(os.path.join(folder, "summary.csv"))
    df["k"] = df.apply(scheme_key, axis=1)
    df["median_ms"] = df["median_s"] * 1e3
    df["q1_ms"], df["q3_ms"] = df["q1_s"] * 1e3, df["q3_s"] * 1e3
    return df


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, "%s.%s" % (name, ext)), dpi=300)
    plt.close(fig)
    print("  wrote figures/%s.png" % name)


def models_of(df):
    return [m for m in MODEL_NAME if m in set(df.model)]


def line(ax, d, x, y, k, err=True, **kw):
    d = d.sort_values(x)
    st = STYLE[k]
    ax.plot(d[x], d[y], color=st["color"], marker=st["marker"], label=st["label"], **kw)
    if err and y == "median_ms":
        ax.fill_between(d[x], d["q1_ms"], d["q3_ms"], color=st["color"], alpha=0.15, lw=0)


# ------------------------------------------------------------------ figures
def fig_time_vs_clients(df, out, W):
    """RQ1: aggregation time vs number of clients, per model."""
    ms = models_of(df)
    fig, axes = plt.subplots(1, len(ms), figsize=(3.2 * len(ms), 3.0), squeeze=False)
    for ax, m in zip(axes[0], ms):
        d = df[df.model == m]
        for k in ["A", "B", "C_slice", "C_clients", "numba"]:
            dk = d[(d.k == k) & ((d.W == W) | (k == "A"))]
            if len(dk):
                line(ax, dk, "N", "median_ms", k)
        n_axis(ax, d.N)
        ax.set_yscale("log")
        ax.set_title(MODEL_NAME[m])
    axes[0][0].set_ylabel("aggregation time per round (ms)")
    fig.suptitle("Aggregation time vs. client count (W = %d workers; band = IQR)" % W, y=1.02)
    legend_below(fig, axes)
    save(fig, out, "fig_rq1_time_vs_clients")


def fig_speedup_vs_clients(df, out, W):
    """RQ1/H1: speedup over Scheme A as N and P grow."""
    ms = models_of(df)
    fig, axes = plt.subplots(1, len(ms), figsize=(3.2 * len(ms), 3.0), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], ms):
        d = df[(df.model == m) & (df.W == W)]
        for k in ["C_slice", "C_clients", "C_e2e", "B", "numba"]:
            if len(d[d.k == k]):
                line(ax, d[d.k == k], "N", "speedup_vs_A", k, err=False)
        ax.axhline(1, color=MUTED, lw=1, ls="--")
        n_axis(ax, df[df.model == m].N)
        ax.set_title(MODEL_NAME[m])
    axes[0][0].set_ylabel("speedup vs. Scheme A (x)")
    fig.suptitle("Speedup over sequential FedAvg (W = %d; dashed = break-even)" % W, y=1.02)
    legend_below(fig, axes)
    save(fig, out, "fig_rq1_speedup_vs_clients")


def fig_copy_cost(df, out, W):
    """RQ2: cost of pickling/copying (B) relative to zero-copy shared memory (C)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    for m in models_of(df):
        d = df[(df.model == m) & (df.W == W)]
        b = d[d.k == "B"].set_index("N")["median_s"]
        c = d[d.k == "C_slice"].set_index("N")["median_s"]
        r = (b / c).dropna()
        mb = d[d.k == "B"].set_index("N")["block_mb"].loc[r.index]
        ax.plot(mb, r.values, marker={"lenet": "o", "mobilenetv2": "s", "resnet18": "^"}[m],
                color={"lenet": "#2a78d6", "mobilenetv2": "#eb6834", "resnet18": "#1baf7a"}[m],
                label=MODEL_NAME[m])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("total update data N x P x 4 B (MB)")
    ax.set_ylabel("time B / time C (x)")
    ax.set_title("Pickling vs. shared memory (W = %d)" % W)
    ax.legend(frameon=False)

    ax = axes[1]   # throughput of each mechanism
    for k in ["A", "B", "C_slice", "C_e2e"]:
        d = df[(df.k == k) & ((df.W == W) | (k == "A"))].copy()
        if not len(d):
            continue
        d["gbps"] = d["block_mb"] / 1024 / d["median_s"]
        d = d.sort_values("block_mb")
        ax.plot(d["block_mb"], d["gbps"], ls="none", marker=STYLE[k]["marker"],
                color=STYLE[k]["color"], label=STYLE[k]["label"])
    ax.set_xscale("log")
    ax.set_xlabel("total update data (MB)")
    ax.set_ylabel("effective throughput (GB/s)")
    ax.set_title("Data processed per second")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_rq2_copy_cost")


def largest_n(df, m):
    return df[df.model == m]["N"].max()


def fig_scaling_workers(df, out):
    """RQ3/H3: speedup and parallel efficiency vs W at the largest N per model."""
    ms = models_of(df)
    fig, axes = plt.subplots(2, len(ms), figsize=(3.2 * len(ms), 5.4), squeeze=False)
    for j, m in enumerate(ms):
        N = largest_n(df, m)
        d = df[(df.model == m) & (df.N == N)]
        Ws = sorted(d[d.k != "A"].W.unique())
        for k in ["C_slice", "C_clients", "C_e2e", "B", "numba"]:
            dk = d[d.k == k]
            if len(dk):
                line(axes[0][j], dk, "W", "speedup_vs_A", k, err=False)
                line(axes[1][j], dk, "W", "efficiency", k, err=False)
        axes[0][j].plot(Ws, Ws, color=MUTED, lw=1, ls=":", label="ideal (linear)")
        axes[1][j].axhline(1, color=MUTED, lw=1, ls=":")
        axes[0][j].set_title("%s, N = %d" % (MODEL_NAME[m], N))
        for i in (0, 1):
            axes[i][j].set_xscale("log", base=2)
            axes[i][j].set_xticks(Ws)
            axes[i][j].set_xticklabels([str(w) for w in Ws])
        axes[1][j].set_xlabel("worker processes W")
    axes[0][0].set_ylabel("speedup vs. A (x)")
    axes[1][0].set_ylabel("parallel efficiency (speedup / W)")
    legend_below(fig, axes)
    fig.subplots_adjust(bottom=0.16, hspace=0.35)
    save(fig, out, "fig_rq3_scaling_workers")


def fig_resources(df, out):
    """RQ3: CPU utilization and memory vs W (largest N of each model)."""
    ms = models_of(df)
    fig, axes = plt.subplots(2, len(ms), figsize=(3.2 * len(ms), 5.4), squeeze=False)
    for j, m in enumerate(ms):
        N = largest_n(df, m)
        d = df[(df.model == m) & (df.N == N)]
        Ws = sorted(d.W.unique())
        for k in ["A", "B", "C_slice", "C_clients", "numba"]:
            dk = d[d.k == k]
            if not len(dk):
                continue
            if k == "A":   # A is W-independent: draw as a reference level
                for i, col in ((0, "proc_cpu_cores"), (1, "peak_mem_mb")):
                    axes[i][j].axhline(dk[col].iloc[0], color=STYLE[k]["color"], lw=1.5,
                                       ls="--", label=STYLE[k]["label"])
                continue
            line(axes[0][j], dk, "W", "proc_cpu_cores", k, err=False)
            line(axes[1][j], dk, "W", "peak_mem_mb", k, err=False)
        axes[0][j].set_title("%s, N = %d (%.0f MB)" % (MODEL_NAME[m], N, d.block_mb.iloc[0]))
        for i in (0, 1):
            axes[i][j].set_xscale("log", base=2)
            axes[i][j].set_xticks(Ws)
            axes[i][j].set_xticklabels([str(w) for w in Ws])
        axes[1][j].set_xlabel("worker processes W")
    axes[0][0].set_ylabel("busy cores (process CPU-s / wall-s)")
    axes[1][0].set_ylabel("peak memory, parent + workers (MB)")
    legend_below(fig, axes)
    fig.subplots_adjust(bottom=0.16, hspace=0.35)
    save(fig, out, "fig_rq3_cpu_memory")


def fig_partitioning(df, out):
    """Slice vs client partitioning of Scheme C."""
    ms = models_of(df)
    fig, axes = plt.subplots(1, len(ms), figsize=(3.2 * len(ms), 3.0), squeeze=False, sharey=True)
    Wcol = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100", 16: "#6250d6"}
    for ax, m in zip(axes[0], ms):
        d = df[df.model == m]
        for W in sorted(d[d.k == "C_slice"].W.unique()):
            s = d[(d.k == "C_slice") & (d.W == W)].set_index("N")["median_s"]
            c = d[(d.k == "C_clients") & (d.W == W)].set_index("N")["median_s"]
            r = (c / s).dropna().sort_index()
            if len(r):
                ax.plot(r.index, r.values, marker="o", color=Wcol.get(W, MUTED), label="W = %d" % W)
        ax.axhline(1, color=MUTED, lw=1, ls="--")
        n_axis(ax, df[df.model == m].N)
        ax.set_title(MODEL_NAME[m])
    axes[0][0].set_ylabel("time(clients) / time(slice)\n> 1: slice partition wins")
    legend_below(fig, axes)
    save(fig, out, "fig_partitioning")


def fig_startup(folder, out):
    p = os.path.join(folder, "startup.csv")
    if not os.path.exists(p):
        return
    s = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.bar(s.W.astype(str), s.startup_s * 1e3, color="#2a78d6", width=0.6)
    for x, v in zip(s.W.astype(str), s.startup_s * 1e3):
        ax.annotate("%.0f" % v, (x, v), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xlabel("worker processes W")
    ax.set_ylabel("pool start-up (ms)")
    ax.set_title("One-time pool start-up (%s)" % s.start_method.iloc[0])
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_pool_startup")


def fig_fl(fl_csv, out):
    d = pd.read_csv(fl_csv)
    sets = list(d.dataset.unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 3.0), squeeze=False)
    fl_style = {"A": "A", "B": "B", "C": "C_slice", "C_clients": "C_clients", "numba": "numba"}
    dash = {"A": "-", "B": "--", "C": ":", "C_clients": "-.", "numba": (0, (1, 3))}
    for ax, ds in zip(axes[0], sets):
        for s in d[d.dataset == ds].scheme.unique():
            x = d[(d.dataset == ds) & (d.scheme == s)].sort_values("round")
            st = STYLE[fl_style[s]]
            ax.plot(x["round"], x["test_acc"] * 100, color=st["color"], ls=dash.get(s, "-"),
                    marker=st["marker"], markevery=max(1, len(x) // 6), label=st["label"])
        ax.set_title({"cifar10": "CIFAR-10 (Dirichlet a=0.5)", "femnist": "FEMNIST"}.get(ds, ds))
        ax.set_xlabel("round")
    axes[0][0].set_ylabel("test accuracy (%)")
    legend_below(fig, axes)
    save(fig, out, "fig_fl_accuracy")


# ------------------------------------------------------------------- tables
def tables(df, folder, fl_csv, Wmax):
    lines = ["# Result tables\n"]
    lines.append("## Aggregation time at W = %d (median ms, speedup vs A)\n" % Wmax)
    lines.append("| Model | N | Data (MB) | A | B | C slice | C clients | C e2e | Numba | "
                 "C slice speedup | B/C |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in models_of(df):
        for N in sorted(df[df.model == m].N.unique()):
            d = df[(df.model == m) & (df.N == N)]

            def t(k):
                x = d[(d.k == k) & ((d.W == Wmax) | (k == "A"))]
                return x.median_ms.iloc[0] if len(x) else np.nan
            a, b, cs = t("A"), t("B"), t("C_slice")
            lines.append("| %s | %d | %.0f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2fx | %.1fx |"
                         % (MODEL_NAME[m], N, d.block_mb.iloc[0], a, b, cs, t("C_clients"),
                            t("C_e2e"), t("numba"), a / cs, b / cs))
    lines.append("\n## Scaling with W at the largest N (speedup / efficiency / busy cores / peak MB)\n")
    lines.append("| Model | N | Scheme | W | median ms | speedup | efficiency | busy cores | "
                 "CPU mean % | peak MB |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in models_of(df):
        N = largest_n(df, m)
        d = df[(df.model == m) & (df.N == N)].sort_values(["k", "W"])
        for _, r in d.iterrows():
            cpu_val = getattr(r, "sys_cpu_mean_pct", getattr(r, "cpu_mean_pct", 0.0))
            lines.append("| %s | %d | %s | %d | %.2f | %.2f | %.2f | %.2f | %.0f | %.0f |"
                         % (MODEL_NAME[m], N, STYLE[r.k]["label"], r.W, r.median_ms,
                            r.speedup_vs_A, r.efficiency, r.proc_cpu_cores, cpu_val,
                            r.peak_mem_mb))
    bad = df[~df.allclose.astype(bool)]
    lines.append("\n## Correctness\n")
    lines.append("All %d scheme/configuration pairs passed allclose(atol=1e-6) against Scheme A; "
                 "largest max-abs difference = %.3g." % (len(df), df.maxdiff_vs_A.max())
                 if not len(bad) else "**%d configurations FAILED allclose** - see summary.csv."
                 % len(bad))
    st = os.path.join(folder, "startup.csv")
    if os.path.exists(st):
        s = pd.read_csv(st)
        lines.append("\n## Worker-pool start-up (%s, excluded from round times)\n" % s.start_method.iloc[0])
        lines.append("| W | start-up (ms) |\n|---|---|")
        for _, r in s.iterrows():
            lines.append("| %d | %.1f |" % (r.W, r.startup_s * 1e3))
    if fl_csv:
        d = pd.read_csv(fl_csv)
        lines.append("\n## Federated training (correctness track)\n")
        lines.append("| Dataset | Scheme | final acc (%) | max abs acc diff vs A (pp) | "
                     "max param diff vs seq | median agg ms |")
        lines.append("|---|---|---|---|---|---|")
        for ds in d.dataset.unique():
            a = d[(d.dataset == ds) & (d.scheme == "A")].set_index("round").test_acc
            for s in d[d.dataset == ds].scheme.unique():
                x = d[(d.dataset == ds) & (d.scheme == s)].set_index("round")
                diff = (x.test_acc - a).abs().max() * 100 if len(a) else np.nan
                lines.append("| %s | %s | %.2f | %.2f | %.3g | %.3f |"
                             % (ds, s, x.test_acc.iloc[-1] * 100, diff,
                                x.maxdiff_vs_seq.max(), x.agg_time_ms.median()))
    path = os.path.join(folder, "tables.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote tables.md")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bench", help="benchmark results folder (contains summary.csv)")
    p.add_argument("--fl", help="fl_*.csv produced by fl_train.py")
    p.add_argument("--W", type=int, help="worker count for the per-N figures (default: max <= 8)")
    a = p.parse_args()
    df = load_bench(a.bench)
    Ws = sorted(df[df.k != "A"].W.unique())
    W = a.W or max([w for w in Ws if w <= 8] or Ws)
    out = os.path.join(a.bench, "figures")
    os.makedirs(out, exist_ok=True)
    fig_time_vs_clients(df, out, W)
    fig_speedup_vs_clients(df, out, W)
    if {"B", "C_slice"} <= set(df.k):
        fig_copy_cost(df, out, W)
    if len(Ws) > 1:
        fig_scaling_workers(df, out)
        fig_resources(df, out)
    if {"C_slice", "C_clients"} <= set(df.k):
        fig_partitioning(df, out)
    fig_startup(a.bench, out)
    if a.fl:
        fig_fl(a.fl, out)
    tables(df, a.bench, a.fl, W)


if __name__ == "__main__":
    main()
