"""Generic plotting functions for the paper figures.

Extracted from ``scripts/make_paper_figures.py`` so the plots can be reused
across scripts and notebooks. Each function is pure: it takes a
pre-shaped ``pandas.DataFrame`` and returns a ``matplotlib.figure.Figure``.
Data loading and file writing stay in the caller (script or notebook).

All figures share one method -> style mapping (``METHOD_STYLES``) so a method
has the same colour, line style and marker in every figure. Figures are drawn
at their printed size (``SINGLE_COL_IN`` / ``DOUBLE_COL_IN``) and should be
included in LaTeX with ``width=\\linewidth`` so fonts print at true size.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from src.vis.style import SINGLE_COL_IN, set_publication_style

# method -> colour for several ColorBrewer palettes. our methods (eIRWR, IRWR)
# always take the most salient pair; baselines are dashed with square
# markers, our methods solid with circles.
PALETTES = {
    # Set2 (soft qualitative): ours in Set2 green/blue, baselines in the
    # remaining Set2 hues (no red, so no method reads as "bad")
    "set2": {
        "MicroRCA": "#E78AC3",
        "MonitorRank": "#FC8D62",
        "CORAL": "#B3B3B3",
        "TraceDiag": "#E5C494",
        "CloudRanger": "#FFD92F",
        "IRWR": "#8DA0CB",
        "eIRWR": "#66C2A5",
    },
    # soft mix: ours in mid Greens/Blues for legibility, baselines in Set3
    # pastels plus BrBG tan and PiYG pink
    "soft_mix": {
        "MicroRCA": "#BC80BD",
        "MonitorRank": "#FDB462",
        "CORAL": "#BEBADA",
        "TraceDiag": "#DFC27D",
        "CloudRanger": "#F1B6DA",
        "IRWR": "#4292C6",
        "eIRWR": "#41AB5D",
    },
    # paired: ours in dark green/blue, baselines in purples, oranges, browns
    "paired": {
        "MicroRCA": "#6A3D9A",
        "MonitorRank": "#FF7F00",
        "CORAL": "#CAB2D6",
        "TraceDiag": "#B15928",
        "CloudRanger": "#FDBF6F",
        "IRWR": "#1F78B4",
        "eIRWR": "#33A02C",
    },
}
DEFAULT_PALETTE = "soft_mix"
_MARKERS = {"IRWR": ("-", "o"), "eIRWR": ("-", "o")}

# method -> (color, linestyle, marker). keys are CSV method names.
METHOD_STYLES: dict[str, tuple[str, str, str]] = {}


def set_palette(name: str = DEFAULT_PALETTE) -> None:
    """Switch every figure to the named palette in ``PALETTES``."""
    for method, color in PALETTES[name].items():
        METHOD_STYLES[method] = (color, *_MARKERS.get(method, ("--", "s")))


set_palette()
GREY = "#999999"  # ColorBrewer Greys
OURS = {"IRWR", "eIRWR"}

# CSV method name -> label used in the paper
DISPLAY = {"CORAL": "Corr-Fusion"}

ACCURACY_METHODS = [
    "MicroRCA",
    "MonitorRank",
    "CORAL",
    "TraceDiag",
    "IRWR",
    "eIRWR",
]
COMPUTE_METHODS = ["MicroRCA", "CloudRanger", "TraceDiag", "IRWR", "eIRWR"]

TOPOLOGIES = [84, 136, 297]
VISIBILITIES = [0.0, 0.1, 0.3, 0.5]

# kept for backward compatibility with older callers
LINE_STYLES = {m: METHOD_STYLES[m] for m in ACCURACY_METHODS}


def _label(method: str) -> str:
    return DISPLAY.get(method, method)


def _line_kwargs(method: str) -> dict:
    """Shared line/marker kwargs so every figure draws a method the same way."""
    color, ls, marker = METHOD_STYLES[method]
    ours = method in OURS
    return {
        "color": color,
        "linestyle": ls,
        "marker": marker,
        "markersize": 2.8 if ours else 2.2,
        "linewidth": 1.0 if ours else 0.7,
        "label": _label(method),
        "zorder": 3 if ours else 2,
    }


def _whiten(fig: plt.Figure, *axes: plt.Axes) -> plt.Figure:
    """Force white backgrounds on the figure and every axis."""
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
    return fig


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="center", pad=2, fontweight="normal")


def bootstrap_mrr_ci(
    per_incident: pd.DataFrame,
    *,
    by: list[str],
    strata: str | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Percentile bootstrap CI of MRR, resampling incidents.

    Parameters
    ----------
    per_incident : DataFrame
        One row per (incident, method, visibility[, topology]) with a
        ``rank`` column (1 = root cause ranked first).
    by : list of str
        Grouping columns, e.g. ``["method", "visibility"]``.
    strata : str, optional
        Column to resample within (e.g. ``"topology"``). The statistic is
        then the unweighted mean of the per-stratum MRRs, matching how the
        paper averages over topologies.

    Returns
    -------
    DataFrame with the ``by`` columns plus ``MRR``, ``lo``, ``hi``.
    """
    rng = np.random.default_rng(seed)
    alpha = (1 - level) / 2
    rows = []
    for key, grp in per_incident.groupby(by, sort=False):
        groups = [g for _, g in grp.groupby(strata)] if strata else [grp]
        point, boots = [], []
        for g in groups:
            rr = 1.0 / g["rank"].to_numpy(dtype=float)
            point.append(rr.mean())
            idx = rng.integers(0, len(rr), size=(n_boot, len(rr)))
            boots.append(rr[idx].mean(axis=1))
        # unweighted mean across strata, per bootstrap replicate
        boot = np.mean(boots, axis=0)
        key = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(by, key, strict=True)),
                "MRR": float(np.mean(point)),
                "lo": float(np.quantile(boot, alpha)),
                "hi": float(np.quantile(boot, 1 - alpha)),
            }
        )
    return pd.DataFrame(rows)


def plot_mrr_vs_visibility(
    mean_df: pd.DataFrame,
    *,
    visibilities: list[float] = VISIBILITIES,
    methods: list[str] = ACCURACY_METHODS,
    figsize: tuple[float, float] = (SINGLE_COL_IN, 1.55),
) -> plt.Figure:
    """MRR vs root-cause visibility, one line per method (single column).

    Parameters
    ----------
    mean_df : DataFrame
        Long-form frame with columns ``method``, ``visibility``, ``MRR``
        (typically averaged over topologies). If it also has ``lo`` and
        ``hi`` columns, a shaded confidence band is drawn per method.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    for method in methods:
        sub = mean_df[
            (mean_df["method"] == method)
            & (mean_df["visibility"].isin(visibilities))
        ].sort_values("visibility")
        kw = _line_kwargs(method)
        ax.plot(sub["visibility"], sub["MRR"], **kw)
        if {"lo", "hi"} <= set(sub.columns):
            ax.fill_between(
                sub["visibility"],
                sub["lo"],
                sub["hi"],
                color=kw["color"],
                alpha=0.2,
                linewidth=0,
                zorder=1,
            )
    ax.set_xlabel("Root-cause visibility ($v$)")
    ax.set_ylabel("MRR")
    ax.set_xticks(visibilities)
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2, loc="upper left")
    return _whiten(fig, ax)


def plot_mrr_by_topology(
    mrr_by_topology: pd.DataFrame,
    *,
    topologies: list[int] = TOPOLOGIES,
    methods: list[str] = ACCURACY_METHODS,
    visibility: float = 0.3,
    ci: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (SINGLE_COL_IN, 1.5),
) -> plt.Figure:
    """Grouped bars: MRR per method across topologies (single column).

    Parameters
    ----------
    mrr_by_topology : DataFrame
        Wide frame indexed by ``method`` with one column per topology id
        (values are the MRR at the chosen visibility).
    ci : DataFrame, optional
        Columns ``method``, ``topology``, ``lo``, ``hi``. If given, each bar
        gets an error bar spanning the interval.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    width = 0.8 / len(methods)
    for k, method in enumerate(methods):
        vals = [mrr_by_topology.loc[method, t] for t in topologies]
        offset = (k - (len(methods) - 1) / 2) * width
        yerr = None
        if ci is not None:
            c = ci[ci["method"] == method].set_index("topology")
            lo = [vals[j] - c.loc[t, "lo"] for j, t in enumerate(topologies)]
            hi = [c.loc[t, "hi"] - vals[j] for j, t in enumerate(topologies)]
            yerr = [lo, hi]
        ax.bar(
            [x + offset for x in range(len(topologies))],
            vals,
            yerr=yerr,
            error_kw={
                "elinewidth": 0.5,
                "capsize": 0.8,
                "capthick": 0.5,
                "ecolor": "#333333",
            },
            width=width,
            color=METHOD_STYLES[method][0],
            edgecolor="white",
            linewidth=0.3,
            label=_label(method),
        )
    ax.set_xticks(range(len(topologies)))
    ax.set_xticklabels([f"CG{i}" for i in topologies])
    ax.set_ylabel(f"MRR at $v={visibility}$")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    # legend sits in the empty band above the bars (MRR < 0.8 everywhere)
    ax.legend(ncol=3, loc="upper center", handlelength=1.0)
    return _whiten(fig, ax)


def plot_runtime_memory_scalability(
    compute_df: pd.DataFrame,
    *,
    methods: list[str] = COMPUTE_METHODS,
    figsize: tuple[float, float] = (SINGLE_COL_IN, 1.8),
) -> plt.Figure:
    """Two-panel runtime and peak-memory vs graph size (single column).

    Parameters
    ----------
    compute_df : DataFrame
        Columns ``method``, ``n_nodes``, ``mean_time_s``, ``peak_memory_mb``.
    """
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for method in methods:
        sub = compute_df[compute_df["method"] == method].sort_values("n_nodes")
        kw = _line_kwargs(method)
        ax1.plot(sub["n_nodes"], sub["mean_time_s"] * 1e3, **kw)
        ax2.plot(sub["n_nodes"], sub["peak_memory_mb"], **kw)
    thousands = FuncFormatter(lambda x, _: f"{x / 1e3:g}")
    for ax, ylabel, tag in (
        (ax1, "Runtime (ms)", "(a)"),
        (ax2, "Peak memory (MB)", "(b)"),
    ):
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([4000, 8000, 12000, 16000])
        ax.xaxis.set_major_formatter(thousands)
        ax.set_xlabel(r"Nodes ($\times 10^3$)")
        ax.set_ylabel(ylabel)
        _panel_label(ax, tag)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="outside upper center",
        handlelength=1.6,
        columnspacing=0.8,
    )
    return _whiten(fig, ax1, ax2)


def plot_iterations_vs_alpha(
    convergence_df: pd.DataFrame,
    *,
    n_nodes: int = 10688,
    methods: tuple[str, ...] = ("IRWR", "eIRWR"),
    default_alpha: float = 0.15,
    ci: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (SINGLE_COL_IN, 1.5),
) -> plt.Figure:
    """Iterations-to-convergence vs restart probability (single column).

    Parameters
    ----------
    convergence_df : DataFrame
        Columns ``method``, ``n_nodes``, ``alpha``, ``mean_iters``,
        ``std_iters``.
    n_nodes : int
        Graph size (row filter) to plot.
    """
    set_publication_style()
    sub = convergence_df[convergence_df["n_nodes"] == n_nodes]
    fig, ax = plt.subplots(figsize=figsize)
    for method in methods:
        m = sub[sub["method"] == method].sort_values("alpha")
        kw = _line_kwargs(method)
        ax.errorbar(
            m["alpha"],
            m["mean_iters"],
            yerr=m["std_iters"],
            capsize=1.5,
            elinewidth=0.5,
            capthick=0.5,
            **kw,
        )
    ax.axvline(default_alpha, color=GREY, linestyle=":", linewidth=0.6)
    ax.annotate(
        rf"$\alpha={default_alpha}$",
        xy=(default_alpha, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        color="0.35",
    )
    ax.set_xlabel(r"Restart probability ($\alpha$)")
    ax.set_ylabel("Iterations to\nconvergence")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    return _whiten(fig, ax)
