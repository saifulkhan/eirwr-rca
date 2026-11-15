"""Visualization package: the figures used in the paper.

Submodules
----------
- ``paper_figures`` : publication figures
- ``style``         : publication-quality matplotlib/seaborn styling
"""

from src.vis.paper_figures import (
    bootstrap_mrr_ci,
    plot_iterations_vs_alpha,
    plot_mrr_by_topology,
    plot_mrr_vs_visibility,
    plot_runtime_memory_scalability,
)
from src.vis.style import set_publication_style

__all__ = [
    # paper_figures
    "bootstrap_mrr_ci",
    "plot_iterations_vs_alpha",
    "plot_mrr_by_topology",
    "plot_mrr_vs_visibility",
    "plot_runtime_memory_scalability",
    # style
    "set_publication_style",
]
