"""
Publication-quality matplotlib and seaborn styling configuration.

This module provides consistent styling for all visualizations in the project,
optimized for academic publications and professional presentations.
"""

import seaborn as sns
from matplotlib import rcParams

# physical figure widths (inches) for Springer SOCA (sn-jnl, iicol layout):
# \columnwidth = 216.24pt, \textwidth = 455.24pt. figures are drawn at these
# exact widths and included at width=\linewidth, so fonts print at true size.
SINGLE_COL_IN = 216.24 / 72.27
DOUBLE_COL_IN = 455.24 / 72.27

# publication-quality settings (sized for a 3 in column, 8 pt captions)
PUBLICATION_SETTINGS = {
    # font settings
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.titlesize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # line and marker settings
    "lines.linewidth": 0.9,
    "lines.markersize": 2.8,
    "patch.linewidth": 0.4,
    # axes settings
    "axes.linewidth": 0.6,
    "axes.labelpad": 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    # grid settings
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "grid.color": "0.8",
    "grid.linestyle": "--",
    # tick settings
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # figure settings: constrained layout and no bbox cropping keep the
    # saved PDF at exactly the requested width
    "figure.dpi": 150,
    "figure.autolayout": False,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.01,
    "figure.constrained_layout.w_pad": 0.01,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.0,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    # legend settings
    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.25,
    "legend.borderaxespad": 0.3,
}


def set_publication_style():
    """Apply publication-quality styling to all matplotlib/seaborn plots."""
    # set seaborn style with white background
    sns.set_style(
        "whitegrid",
        {
            "axes.edgecolor": "0.2",
            "grid.color": "0.8",
            "grid.linestyle": "--",
        },
    )

    # apply custom rcParams last so seaborn's context does not override them
    rcParams.update(PUBLICATION_SETTINGS)


def save_figure(fig, path, dpi=300, transparent=False, **kwargs):
    """
    Save figure with publication-quality settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Output file path.
    dpi : int, default=300
        Resolution for raster formats.
    transparent : bool, default=False
        Whether to use transparent background.
    **kwargs
        Additional arguments passed to fig.savefig().
    """
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="white" if not transparent else "none",
        edgecolor="none",
        transparent=transparent,
        **kwargs,
    )
