"""
Publication-quality matplotlib and seaborn styling configuration.

This module provides consistent styling for all visualizations in the project,
optimized for academic publications and professional presentations.
"""

import seaborn as sns
from matplotlib import rcParams

# Publication-quality settings
PUBLICATION_SETTINGS = {
    # Font settings
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    # Line and marker settings
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "patch.linewidth": 0.5,
    # Axes settings
    "axes.linewidth": 0.8,
    "axes.labelpad": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    # Grid settings
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "grid.color": "0.8",
    "grid.linestyle": "--",
    # Tick settings
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Figure settings
    "figure.dpi": 100,
    "figure.autolayout": False,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    # Legend settings
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
}


def set_publication_style():
    """Apply publication-quality styling to all matplotlib/seaborn plots."""
    # Set seaborn style with white background
    sns.set_style(
        "whitegrid",
        {
            "axes.edgecolor": "0.2",
            "grid.color": "0.8",
            "grid.linestyle": "--",
        },
    )

    # Apply custom rcParams
    rcParams.update(PUBLICATION_SETTINGS)

    # Set seaborn context for better scaling
    sns.set_context(
        "paper",
        font_scale=1.0,
        rc={
            "lines.linewidth": 1.5,
            "axes.linewidth": 0.8,
        },
    )


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
