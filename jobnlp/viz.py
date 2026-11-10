"""Shared plotting style for the notebook figures (matplotlib / seaborn).

Palette: validated 8-slot categorical palette (CVD-checked, fixed order, never cycled),
a one-hue blue sequential ramp for magnitudes, and recessive grey axes/grids."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "<", ">"]
PRIMARY = "#2a78d6"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
NODE_COLORS = {"company": CATEGORICAL[1], "job": CATEGORICAL[0], "skill": CATEGORICAL[2]}


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlepad": 12,
        "axes.labelsize": 10.5, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "legend.frameon": False, "legend.fontsize": 9.5,
        "font.family": "DejaVu Sans", "figure.dpi": 110, "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def cluster_color(i: int) -> str:
    return CATEGORICAL[i % len(CATEGORICAL)]


def cluster_marker(i: int) -> str:
    # beyond 8 clusters the colour repeats, so the marker shape always differs
    return MARKERS[i % len(MARKERS)]


def hbar(ax, labels, values, title, xlabel, fmt="{:,.0f}", color=PRIMARY, sublabels=None):
    """Horizontal bar chart, largest at top, direct value labels, recessive axes."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color, height=0.68, zorder=2)
    ax.set_yticks(y, [f"{l}" if sublabels is None else f"{l}  ·  {s}" for l, s in
                      zip(labels, sublabels or [None] * len(labels))])
    ax.set_title(title, loc="left"); ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    vmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.01, yi, fmt.format(v), va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, vmax * 1.12)
    ax.tick_params(axis="y", length=0)
    return ax


def save(fig, fig_dir: Path, name: str):
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")


def spread_labels(ys, min_gap):
    """1-D label de-overlap: returns positions >= min_gap apart, as close as possible to ys."""
    order = np.argsort(ys)
    pos = np.array(ys, dtype=float)[order]
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < min_gap:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = mid - min_gap / 2, mid + min_gap / 2
                moved = True
        if not moved:
            break
    out = np.empty_like(pos); out[order] = pos
    return out
