"""Run pairwise Wilcoxon eIRWR vs each baseline from per-incident ranks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rca.statistics import compute_effect_sizes, pairwise_wilcoxon  # noqa: E402


def main() -> None:
    per_incident = PROJECT_ROOT / "results" / "per_incident_ranks.csv"
    if not per_incident.exists():
        print(f"ERROR: {per_incident} not found")
        sys.exit(1)

    df = pd.read_csv(per_incident)
    print(f"Loaded {len(df):,} rows; methods={df['method'].unique().tolist()}")

    all_methods = df["method"].unique().tolist()
    proposed = "eIRWR"
    baselines = [m for m in all_methods if m != proposed]

    w = pairwise_wilcoxon(df, proposed, baselines)
    w["proposed"] = proposed
    out = PROJECT_ROOT / "results" / "statistical_wilcoxon.csv"
    w.to_csv(out, index=False)
    print(f"Saved Wilcoxon -> {out} ({len(w)} rows)")

    e = compute_effect_sizes(df, proposed, baselines)
    e["proposed"] = proposed
    out2 = PROJECT_ROOT / "results" / "statistical_effect_sizes.csv"
    e.to_csv(out2, index=False)
    print(f"Saved effect sizes -> {out2} ({len(e)} rows)")


if __name__ == "__main__":
    main()
