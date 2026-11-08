"""
Shared plotting helpers and styling constants for benchmark and paper figures.
"""
import copy
import os
import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

# Fixed colour + marker per scheme (colour follows the entity, never its rank).
STYLE = {
    "A":         dict(color="#2a78d6", marker="o", label="A: sequential"),
    "B":         dict(color="#eb6834", marker="s", label="B: multiprocessing + pickling"),
    "C_slice":   dict(color="#1baf7a", marker="^", label="C: shared memory (slice)"),
    "C_clients": dict(color="#eda100", marker="v", label="C: shared memory (clients)"),
    "C_e2e":     dict(color="#e87ba4", marker="D", label="C: slice, incl. write to shm"),
    "numba":     dict(color="#6250d6", marker="P", label="Numba threads [6]"),
}

MODEL_NAME = {
    "lenet": "LeNet (62K)",
    "mobilenetv2": "MobileNetV2 (3.5M)",
    "resnet18": "ResNet-18 (11.2M)",
}

MODELS = ["lenet", "mobilenetv2", "resnet18"]
MCOL = {"lenet": "#2a78d6", "mobilenetv2": "#eb6834", "resnet18": "#1baf7a"}
MMARK = {"lenet": "o", "mobilenetv2": "s", "resnet18": "^"}

INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def scheme_key(row):
    """Normalize (scheme, partition) to the canonical scheme key."""
    s, p = row["scheme"], row["partition"]
    return {"A_sequential": "A", "B_pickle": "B", "numba": "numba",
            "C_shm_e2e": "C_e2e"}.get(s, "C_" + p if s == "C_shm" else s)


def n_axis(ax, Ns):
    """Log x-axis labelled with the actual client counts."""
    Ns = sorted(set(int(n) for n in Ns))
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("clients N")


def plain_log(ax, axis="y"):
    """Format log axis numbers plainly as integers or floats without scientific notation."""
    f = matplotlib.ticker.FuncFormatter(lambda v, _: ("%g" % v))
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(f)
    a.set_minor_formatter(matplotlib.ticker.NullFormatter())


def legend_top(fig, axes, ncol):
    """Deduplicated horizontal legend positioned at the top of the figure."""
    h, l = [], []
    for ax in np.ravel(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.02),
               handlelength=2.2, columnspacing=1.2)


def legend_below(fig, axes):
    """Deduplicated horizontal legend positioned at the bottom of the figure."""
    h, l = [], []
    for ax in np.ravel(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=min(3, len(l)), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.28)
