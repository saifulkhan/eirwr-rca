"""Visualization package — the figures used in ``paper/main.tex``.

Submodules
----------
- ``paper_figures`` : Publication figures for paper/main.tex
- ``style``         : Publication-quality matplotlib/seaborn styling
"""

from src.vis.paper_figures import (
    plot_iterations_vs_alpha,
    plot_mrr_by_topology,
    plot_mrr_vs_visibility,
    plot_runtime_memory_scalability,
)
from src.vis.style import save_figure, set_publication_style

__all__ = [
    # paper_figures
    "plot_iterations_vs_alpha",
    "plot_mrr_by_topology",
    "plot_mrr_vs_visibility",
    "plot_runtime_memory_scalability",
    # style
    "save_figure",
    "set_publication_style",
]
