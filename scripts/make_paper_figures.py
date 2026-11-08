"""Regenerate the figures included in paper/main.tex from results/*.csv.

Produces (PDF + PNG) in paper/figures/:
  - mrr_vs_visibility     (main text, Fig. mrr-visibility)
  - mrr_per_topology      (appendix, Fig. per-topology)
  - scalability           (main text, Fig. scalability)
  - convergence_vs_alpha  (appendix, Fig. convergence)

The plotting logic lives in ``src.vis.paper_figures`` (pure functions that
take DataFrames and return Figures). This script only loads the CSVs, calls
those functions, and writes the files.
See ``notebooks/make_paper_figures.ipynb`` for an interactive version.

Usage:  uv run python scripts/make_paper_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vis.paper_figures import (  # noqa: E402
    TOPOLOGIES,
    bootstrap_mrr_ci,
    plot_iterations_vs_alpha,
    plot_mrr_by_topology,
    plot_mrr_vs_visibility,
    plot_runtime_memory_scalability,
)
from src.vis.style import set_publication_style  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"


def load_mean_comparison() -> pd.DataFrame:
    """Unweighted mean of the three per-topology comparison CSVs
    (this is the aggregation used for the paper's main table)."""
    frames = [
        pd.read_csv(RESULTS / f"full_comparison_metrics_CG{i}.csv")
        for i in TOPOLOGIES
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.groupby(["visibility", "method"], as_index=False)[
        ["PR@1", "PR@3", "PR@5", "MRR", "avg_rank"]
    ].mean()


def load_mrr_by_topology(visibility: float = 0.3) -> pd.DataFrame:
    """Wide frame indexed by method, one column per topology id, MRR values."""
    cols = {}
    for i in TOPOLOGIES:
        df = pd.read_csv(RESULTS / f"full_comparison_metrics_CG{i}.csv")
        cols[i] = df[df["visibility"] == visibility].set_index("method")["MRR"]
    return pd.DataFrame(cols)


def load_per_incident() -> pd.DataFrame:
    """Per-incident ranks for all topologies, with a ``topology`` column."""
    frames = []
    for i in TOPOLOGIES:
        df = pd.read_csv(RESULTS / f"per_incident_ranks_CG{i}.csv")
        df["topology"] = i
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)
    print(f"wrote {FIGURES / stem}.pdf")


def main() -> None:
    set_publication_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    per_incident = load_per_incident()

    # 95% bootstrap CI over incidents, resampled within each topology
    ci_vis = bootstrap_mrr_ci(
        per_incident, by=["method", "visibility"], strata="topology"
    )
    mean_df = load_mean_comparison().merge(
        ci_vis[["method", "visibility", "lo", "hi"]],
        on=["method", "visibility"],
        how="left",
    )
    save(plot_mrr_vs_visibility(mean_df), "mrr_vs_visibility")

    ci_topo = bootstrap_mrr_ci(
        per_incident[per_incident["visibility"] == 0.3],
        by=["method", "topology"],
    )
    save(
        plot_mrr_by_topology(load_mrr_by_topology(), ci=ci_topo),
        "mrr_per_topology",
    )
    save(
        plot_runtime_memory_scalability(
            pd.read_csv(RESULTS / "computational_metrics.csv")
        ),
        "scalability",
    )
    save(
        plot_iterations_vs_alpha(
            pd.read_csv(RESULTS / "convergence_metrics.csv")
        ),
        "convergence_vs_alpha",
    )


if __name__ == "__main__":
    main()
