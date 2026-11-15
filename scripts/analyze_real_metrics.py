"""Detect real anomaly windows in the MSRTMCR response-time data.

Reads the MSRTMCR file with the same index as the topology, so the
metrics cover the same three minutes as its call graph. A reading is
anomalous when the response time of a service exceeds its mean by more
than ``--z-threshold`` standard deviations (computed over the whole
file). A timestamp is kept when at least three services are anomalous
at the same time. The result is the input to the real-anomaly
evaluation (``evaluate_performance.py --experiments real``).

Usage:
    uv run python scripts/analyze_real_metrics.py --dataset-index 297
Writes: results/real_anomaly_windows.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import load_msrt_mcr  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"


def identify_anomaly_windows(
    msrtmcr_df: pd.DataFrame,
    rt_col: str = "providerrpc_rt",
    z_threshold: float = 3.0,
    min_services_affected: int = 3,
) -> pd.DataFrame:
    """Detect anomaly windows using z-score on response time.

    For each timestamp window, flags services where RT exceeds
    mean + z * std computed over the full time series.

    Returns DataFrame of anomaly events with columns:
    timestamp, msname, rt_value, z_score, baseline_mean, baseline_std.
    """
    valid = msrtmcr_df.dropna(subset=[rt_col])
    if len(valid) == 0:
        return pd.DataFrame()

    # per-service baselines
    baselines = (
        valid.groupby("msname")[rt_col].agg(["mean", "std"]).reset_index()
    )
    baselines.columns = ["msname", "baseline_mean", "baseline_std"]
    baselines["baseline_std"] = baselines["baseline_std"].replace(0, np.nan)

    merged = valid[["timestamp", "msname", rt_col]].merge(
        baselines, on="msname"
    )
    merged["z_score"] = (merged[rt_col] - merged["baseline_mean"]) / merged[
        "baseline_std"
    ]

    # filter to anomalous readings
    anomalous = merged[merged["z_score"] > z_threshold].copy()
    anomalous = anomalous.rename(columns={rt_col: "rt_value"})

    # require multiple services affected at same timestamp for systemic anomaly
    ts_counts = anomalous.groupby("timestamp")["msname"].nunique()
    systemic_ts = ts_counts[ts_counts >= min_services_affected].index
    anomalous = anomalous[anomalous["timestamp"].isin(systemic_ts)]

    return anomalous.sort_values(
        ["timestamp", "z_score"], ascending=[True, False]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=297,
        help="Topology index; reads MCRRTUpdate_<index>.csv (default: 297)",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for anomaly detection (default: 3.0)",
    )
    args = parser.parse_args()

    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading MSRTMCR file {args.dataset_index}...")
    msrtmcr_df = load_msrt_mcr(args.dataset_index)
    print(f"  {len(msrtmcr_df):,} rows")

    print(f"Detecting anomaly windows (z > {args.z_threshold})...")
    anomalies = identify_anomaly_windows(
        msrtmcr_df, z_threshold=args.z_threshold
    )
    out = RESULTS_DIR / "real_anomaly_windows.csv"
    anomalies.to_csv(out, index=False)
    n_ts = anomalies["timestamp"].nunique() if len(anomalies) > 0 else 0
    n_svc = anomalies["msname"].nunique() if len(anomalies) > 0 else 0
    print(
        f"  Found {len(anomalies)} anomalous readings "
        f"across {n_ts} timestamps, {n_svc} services"
    )
    print(f"  Saved to {out}")
    print(f"\nCompleted in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
