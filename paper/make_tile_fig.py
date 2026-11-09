"""
Tile-size sweep figure: median aggregation time of Scheme C (slice) against the
inner tile size, one line per worker count.

The point of the figure is that the optimal tile is not a constant: it moves
left as workers are added, because the W workers share one last-level cache and
each holds two tile-sized buffers. The dashed vertical line marks the
conventional fixed 1 MB tile, which is optimal at W=1 and increasingly wrong
beyond it.

    python paper/make_tile_fig.py <sweep_root>

<sweep_root> contains one subdirectory per tile size, named t<KB> (as produced
by sweeping FL_AGG_TILE_KB), each holding benchmark.py output folders.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_common import INK, MUTED, GRID, MODEL_NAME  # noqa: E402

WCOL = {1: "#2a78d6", 2: "#1baf7a", 4: "#eda100", 8: "#eb6834"}
WMARK = {1: "o", 2: "^", 4: "s", 8: "D"}


def collect(root):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "t*"))):
        try:
            tile = int(os.path.basename(d)[1:])
        except ValueError:
            continue
        for s in glob.glob(os.path.join(d, "*", "summary.csv")):
            df = pd.read_csv(s)
            for r in df.itertuples():
                if r.scheme == "C_shm" and r.partition == "slice":
                    rows.append(dict(tile=tile, model=r.model, N=r.N, W=int(r.W),
                                     ms=r.median_s * 1000))
                elif r.scheme == "A_sequential":
                    rows.append(dict(tile=tile, model=r.model, N=r.N, W=0,
                                     ms=r.median_s * 1000))
    return pd.DataFrame(rows)


def main():
    root = sys.argv[1]
    d = collect(root)
    if d.empty:
        sys.exit("no data found under " + root)
    models = [m for m in ["mobilenetv2", "resnet18"] if m in set(d.model)]

    fig, axes = plt.subplots(1, len(models), figsize=(7.0, 2.9), sharey=False)
    if len(models) == 1:
        axes = [axes]

    for ax, m in zip(axes, models):
        sub = d[d.model == m]
        N = sorted(sub.N.unique())[-1]
        sub = sub[sub.N == N]
        for W in sorted(w for w in sub.W.unique() if w > 0):
            s = sub[sub.W == W].sort_values("tile")
            ax.plot(s.tile, s.ms, marker=WMARK.get(W, "o"), ms=4, lw=1.6,
                    color=WCOL.get(W, MUTED), label=f"$W$={W}")
        a = sub[sub.W == 0].sort_values("tile")
        if len(a):
            # Fair baseline: Scheme A at ITS OWN best tile, not its mean over
            # tiles. A is tiled too, and a badly tiled baseline would flatter
            # every parallel curve above it.
            abest = a.ms.min()
            ax.axhline(abest, ls=":", lw=1.3, color=INK, zorder=1)
            ax.text(a.tile.min(), abest, " sequential (best tile) ", va="bottom",
                    ha="left", fontsize=6.5, color=INK)
        ax.axvline(1024, ls="--", lw=1.1, color=MUTED, zorder=0)
        ax.text(1024, ax.get_ylim()[1], " fixed 1 MB ", rotation=90,
                va="top", ha="right", fontsize=6.5, color=MUTED)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub.tile.unique()))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, p: f"{int(v)}" if v < 1024 else f"{int(v)//1024}M"))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(f"{MODEL_NAME.get(m, m)}  ($N$={N})", fontsize=8.5)
        ax.set_xlabel("inner tile size (KB)", fontsize=8)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("aggregation time (ms)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "fig_tile." + ext), dpi=200,
                    bbox_inches="tight")
    print("wrote", os.path.join(out, "fig_tile.{pdf,png}"))

    print("\noptimal tile per W (median ms):")
    for m in models:
        sub = d[(d.model == m)]
        N = sorted(sub.N.unique())[-1]
        sub = sub[sub.N == N]
        for W in sorted(w for w in sub.W.unique() if w > 0):
            s = sub[sub.W == W].sort_values("tile")
            b = s.loc[s.ms.idxmin()]
            at1m = s[s.tile == 1024].ms
            pen = (float(at1m.iloc[0]) / b.ms - 1) * 100 if len(at1m) else float("nan")
            print(f"  {m:12s} W={W}: best tile {int(b.tile):5d} KB "
                  f"({b.ms:7.2f} ms); fixed 1 MB costs {pen:+5.1f}%")


if __name__ == "__main__":
    main()
