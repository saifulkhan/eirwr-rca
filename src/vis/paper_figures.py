"""Generic plotting functions for the paper figures.

Extracted from ``scripts/make_paper_figures.py`` so the plots can be reused
across scripts and notebooks. Each function is pure: it takes a
pre-shaped ``pandas.DataFrame`` and returns a ``matplotlib.figure.Figure``.
Data loading and file writing stay in the caller (script or notebook).

The hard-coded colours / line styles below match the published figures;
keep them stable so regenerated PDFs stay byte-comparable.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.vis.style import set_publication_style


def _whiten(fig: plt.Figure, *axes: plt.Axes) -> plt.Figure:
    """Force white backgrounds on the figure and every axis.

    seaborn's ``whitegrid`` already whitens the axes, but this makes the
    publication (white-background) look explicit and caller-independent.
    """
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
    return fig

# Method -> (color, linestyle, marker); ours are solid, baselines dashed.
# Keys are CSV method names; DISPLAY maps them to the paper's labels.
# Colours are ColorBrewer Pastel1 hues pushed halfway to Set1 for a softer-
# but-saturated look; IRWR/eIRWR take the warm orange/red so our methods
# stay the salient pair against the baselines.
LINE_STYLES = {
    "MicroRCA": ("#BB8CC3", "--", "s"),  # purple
    "MonitorRank": ("#75A5CD", "--", "s"),  # blue
    "CORAL": ("#8CCD87", "--", "s"),  # green
    "TraceDiag": ("#C59772", "--", "s"),  # tan
    "IRWR": ("#FEAC53", "-", "o"),  # orange
    "eIRWR": ("#EF6765", "-", "o"),  # red
}
DISPLAY = {"CORAL": "Corr-Fusion"}
COMPUTE_METHODS = ["MicroRCA", "CloudRanger", "TraceDiag", "IRWR", "eIRWR"]
COMPUTE_COLORS = ["#BB8CC3", "#75A5CD", "#C59772", "#FEAC53", "#EF6765"]

TOPOLOGIES = [84, 136, 297]
VISIBILITIES = [0.0, 0.1, 0.3, 0.5]


def plot_mrr_vs_visibility(
    mean_df: pd.DataFrame,
    *,
    visibilities: list[float] = VISIBILITIES,
    line_styles: dict = LINE_STYLES,
    display: dict = DISPLAY,
) -> plt.Figure:
    """MRR vs root-cause visibility, one line per method.

    Parameters
    ----------
    mean_df : DataFrame
        Long-form frame with columns ``method``, ``visibility``, ``MRR``
        (typically averaged over topologies).
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for method, (color, ls, marker) in line_styles.items():
        sub = mean_df[
            (mean_df["method"] == method)
            & (mean_df["visibility"].isin(visibilities))
        ].sort_values("visibility")
        ax.plot(
            sub["visibility"],
            sub["MRR"],
            ls,
            color=color,
            marker=marker,
            markersize=7 if ls == "-" else 4,
            linewidth=2.5 if ls == "-" else 1.5,
            label=display.get(method, method),
        )
    ax.set_xlabel("Root-Cause Visibility ($v$)")
    ax.set_ylabel("MRR")
    # ax.set_title("Mean Reciprocal Rank vs Root-Cause Visibility\n"
    #             "(averaged over 3 topologies)")
    ax.set_xticks(visibilities)
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=2, loc="upper left")
    return _whiten(fig, ax)


def plot_mrr_by_topology(
    mrr_by_topology: pd.DataFrame,
    *,
    topologies: list[int] = TOPOLOGIES,
    line_styles: dict = LINE_STYLES,
    display: dict = DISPLAY,
) -> plt.Figure:
    """Grouped bars: MRR per method across topologies (at a fixed visibility).

    Parameters
    ----------
    mrr_by_topology : DataFrame
        Wide frame indexed by ``method`` with one column per topology id
        (values are the MRR at the chosen visibility).
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    methods = list(line_styles)
    width = 0.13
    for k, method in enumerate(methods):
        vals = [mrr_by_topology.loc[method, t] for t in topologies]
        offset = (k - (len(methods) - 1) / 2) * width
        ax.bar(
            [x + offset for x in range(len(topologies))],
            vals,
            width=width,
            color=line_styles[method][0],
            label=display.get(method, method),
        )
    ax.set_xticks(range(len(topologies)))
    ax.set_xticklabels([f"CG{i}" for i in topologies])
    ax.set_ylabel("MRR at $v=0.3$")
    # ax.set_title("MRR by Method Across Topologies ($v = 0.3$)")
    ax.legend(
        ncol=3,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )
    return _whiten(fig, ax)


def plot_runtime_memory_scalability(
    compute_df: pd.DataFrame,
    *,
    methods: list[str] = COMPUTE_METHODS,
    colors: list[str] = COMPUTE_COLORS,
) -> plt.Figure:
    """Two-panel runtime and peak-memory vs graph size (log-scale y).

    Parameters
    ----------
    compute_df : DataFrame
        Columns ``method``, ``n_nodes``, ``mean_time_s``, ``peak_memory_mb``.
    """
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for method, color in zip(methods, colors, strict=True):
        sub = compute_df[compute_df["method"] == method].sort_values("n_nodes")
        ax1.plot(
            sub["n_nodes"],
            sub["mean_time_s"] * 1e3,
            "-o",
            color=color,
            label=method,
        )
        ax2.plot(
            sub["n_nodes"],
            sub["peak_memory_mb"],
            "-o",
            color=color,
            label=method,
        )
    for ax, ylabel, title in (
        (ax1, "Runtime (ms)", "(a) Runtime vs Graph Size"),
        (ax2, "Peak Memory (MB)", "(b) Peak Memory vs Graph Size"),
    ):
        ax.set_yscale("log")
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel(ylabel)
        # ax.set_title(title)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(labels),
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _whiten(fig, ax1, ax2)


def plot_iterations_vs_alpha(
    convergence_df: pd.DataFrame,
    *,
    n_nodes: int = 10688,
    methods: tuple[tuple[str, str], ...] = (
        ("IRWR", "#FEAC53"),
        ("eIRWR", "#EF6765"),
    ),
) -> plt.Figure:
    """Iterations-to-convergence vs restart probability for one graph size.

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
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for method, color in methods:
        m = sub[sub["method"] == method].sort_values("alpha")
        ax.errorbar(
            m["alpha"],
            m["mean_iters"],
            yerr=m["std_iters"],
            fmt="-o",
            color=color,
            capsize=3,
            label=method,
        )
    ax.axvline(0.15, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel(r"Restart Probability ($\alpha$)")
    ax.set_ylabel("Iterations to Convergence")
    # ax.set_title(f"Convergence vs Restart Probability ($N$ = {n_nodes:,})")
    ax.legend()
    return _whiten(fig, ax)
