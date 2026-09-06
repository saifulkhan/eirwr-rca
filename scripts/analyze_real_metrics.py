"""Analyze real metric distributions for calibrating anomaly simulation.

Processes MSMetrics and MSRTMCR data from the Alibaba Microservice Trace
dataset to extract per-service metric distributions and detect real
anomaly windows.

Usage:
    python scripts/analyze_real_metrics.py
    python scripts/analyze_real_metrics.py --msrtmcr-index 0 --nrows 1000000
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

from src.loader import load_call_graph_by_index, load_ms_metrics, load_msrt_mcr

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_ROOT = PROJECT_ROOT / "data" / "data"


# ── Per-service metric distributions ───────────────────────────────────


def compute_service_rt_distributions(
    msrtmcr_df: pd.DataFrame,
    rt_col: str = "providerrpc_rt",
) -> pd.DataFrame:
    """Compute per-service response time statistics.

    Returns DataFrame indexed by msname with columns:
    mean_rt, std_rt, cv_rt, p50_rt, p95_rt, p99_rt, n_samples.
    """
    valid = msrtmcr_df.dropna(subset=[rt_col])
    if len(valid) == 0:
        return pd.DataFrame()

    stats = valid.groupby("msname")[rt_col].agg(
        mean_rt="mean",
        std_rt="std",
        p50_rt="median",
        n_samples="count",
    )
    # Quantiles separately
    q95 = valid.groupby("msname")[rt_col].quantile(0.95).rename("p95_rt")
    q99 = valid.groupby("msname")[rt_col].quantile(0.99).rename("p99_rt")
    stats = stats.join(q95).join(q99)

    stats["cv_rt"] = stats["std_rt"] / stats["mean_rt"].replace(0, np.nan)
    stats["p99_p50_ratio"] = stats["p99_rt"] / stats["p50_rt"].replace(0, np.nan)

    return stats.reset_index()


def compute_service_cpu_distributions(
    msmetrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-service CPU/memory statistics."""
    stats = msmetrics_df.groupby("msname").agg(
        mean_cpu=("cpu_utilization", "mean"),
        std_cpu=("cpu_utilization", "std"),
        max_cpu=("cpu_utilization", "max"),
        mean_mem=("memory_utilization", "mean"),
        std_mem=("memory_utilization", "std"),
        max_mem=("memory_utilization", "max"),
        n_samples=("cpu_utilization", "count"),
    )
    stats["cv_cpu"] = stats["std_cpu"] / stats["mean_cpu"].replace(0, np.nan)
    return stats.reset_index()


# ── Anomaly window detection ───────────────────────────────────────────


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

    # Per-service baselines
    baselines = valid.groupby("msname")[rt_col].agg(["mean", "std"]).reset_index()
    baselines.columns = ["msname", "baseline_mean", "baseline_std"]
    baselines["baseline_std"] = baselines["baseline_std"].replace(0, np.nan)

    merged = valid[["timestamp", "msname", rt_col]].merge(baselines, on="msname")
    merged["z_score"] = (
        (merged[rt_col] - merged["baseline_mean"]) / merged["baseline_std"]
    )

    # Filter to anomalous readings
    anomalous = merged[merged["z_score"] > z_threshold].copy()
    anomalous = anomalous.rename(columns={rt_col: "rt_value"})

    # Require multiple services affected at same timestamp for systemic anomaly
    ts_counts = anomalous.groupby("timestamp")["msname"].nunique()
    systemic_ts = ts_counts[ts_counts >= min_services_affected].index
    anomalous = anomalous[anomalous["timestamp"].isin(systemic_ts)]

    return anomalous.sort_values(["timestamp", "z_score"], ascending=[True, False])


def find_anomaly_cascades(
    anomaly_windows: pd.DataFrame,
    call_graph_df: pd.DataFrame,
    lag_minutes: int = 3,
) -> pd.DataFrame:
    """Find temporally correlated anomaly cascades along call graph edges.

    For each anomaly at service A, checks if callees of A also show
    anomalies within a lag window.
    """
    if len(anomaly_windows) == 0:
        return pd.DataFrame()

    # Build edge set from call graph
    edges = set(zip(call_graph_df["um"], call_graph_df["dm"]))

    lag_ms = lag_minutes * 60 * 1000  # convert to milliseconds (timestamp unit)

    cascades = []
    ts_groups = anomaly_windows.groupby("timestamp")

    timestamps = sorted(anomaly_windows["timestamp"].unique())
    for i, ts in enumerate(timestamps):
        current = ts_groups.get_group(ts)
        current_services = set(current["msname"])

        # Look ahead in next timestamps within lag window
        downstream_services = set()
        for j in range(i + 1, len(timestamps)):
            if timestamps[j] - ts > lag_ms:
                break
            future = ts_groups.get_group(timestamps[j])
            downstream_services.update(future["msname"])

        # Check for cascade: caller anomalous now, callee anomalous soon after
        for svc in current_services:
            for ds_svc in downstream_services:
                if (svc, ds_svc) in edges:
                    cascades.append({
                        "caller": svc,
                        "callee": ds_svc,
                        "caller_timestamp": ts,
                        "lag_window_ms": lag_ms,
                    })

    return pd.DataFrame(cascades)


# ── Service overlap analysis ───────────────────────────────────────────


def compute_service_overlap(
    call_graph_df: pd.DataFrame,
    msmetrics_df: pd.DataFrame | None = None,
    msrtmcr_df: pd.DataFrame | None = None,
) -> dict[str, set[str]]:
    """Compute service overlap across datasets."""
    cg_services = (
        set(call_graph_df["um"].dropna()) | set(call_graph_df["dm"].dropna())
    ) - {"UNKNOWN"}

    result = {"callgraph": cg_services}

    if msmetrics_df is not None:
        ms_services = set(msmetrics_df["msname"].dropna())
        result["msmetrics"] = ms_services
        result["cg_and_ms"] = cg_services & ms_services

    if msrtmcr_df is not None:
        rt_services = set(msrtmcr_df["msname"].dropna())
        result["msrtmcr"] = rt_services
        result["cg_and_rt"] = cg_services & rt_services

    if msmetrics_df is not None and msrtmcr_df is not None:
        result["all_three"] = cg_services & ms_services & rt_services

    return result


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analyze real metric distributions"
    )
    parser.add_argument(
        "--call-graph-index", type=int, default=297,
        help="CallGraph index for topology (default: 297)",
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Limit rows read from MSRTMCR (default: all)",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=3.0,
        help="Z-score threshold for anomaly detection (default: 3.0)",
    )
    args = parser.parse_args()

    total_t0 = time.time()

    # Load CallGraph for topology
    print("Loading CallGraph...")
    cg_df = load_call_graph_by_index(args.call_graph_index, nrows=args.nrows)
    print(f"  {len(cg_df):,} rows")

    # Load MSRTMCR for response time metrics
    print("Loading MSRTMCR...")
    msrtmcr_path = DATA_ROOT / "MSRTMCR" / "MCRRTUpdate_0.csv"
    if msrtmcr_path.exists():
        msrtmcr_df = load_msrt_mcr(path=msrtmcr_path, nrows=args.nrows)
        print(f"  {len(msrtmcr_df):,} rows")
    else:
        print(f"  WARNING: {msrtmcr_path} not found, skipping MSRTMCR analysis")
        msrtmcr_df = None

    # Load MSMetrics
    print("Loading MSMetrics...")
    msmetrics_path = DATA_ROOT / "MSMetrics" / "MSMetricsUpdate_0.csv"
    if msmetrics_path.exists():
        msmetrics_df = load_ms_metrics(path=msmetrics_path, nrows=args.nrows)
        print(f"  {len(msmetrics_df):,} rows")
    else:
        print(f"  WARNING: {msmetrics_path} not found, skipping MSMetrics analysis")
        msmetrics_df = None

    # Service overlap
    print("\nComputing service overlap...")
    overlap = compute_service_overlap(cg_df, msmetrics_df, msrtmcr_df)
    for key, svcs in overlap.items():
        print(f"  {key}: {len(svcs):,} services")

    # RT distributions
    if msrtmcr_df is not None:
        print("\nComputing RT distributions...")
        rt_stats = compute_service_rt_distributions(msrtmcr_df)
        out = RESULTS_DIR / "rt_distributions.csv"
        rt_stats.to_csv(out, index=False)
        print(f"  Saved {len(rt_stats)} service stats to {out}")

    # CPU distributions
    if msmetrics_df is not None:
        print("\nComputing CPU/memory distributions...")
        cpu_stats = compute_service_cpu_distributions(msmetrics_df)
        out = RESULTS_DIR / "cpu_distributions.csv"
        cpu_stats.to_csv(out, index=False)
        print(f"  Saved {len(cpu_stats)} service stats to {out}")

    # Anomaly windows
    if msrtmcr_df is not None:
        print(f"\nDetecting anomaly windows (z > {args.z_threshold})...")
        anomalies = identify_anomaly_windows(
            msrtmcr_df, z_threshold=args.z_threshold,
        )
        out = RESULTS_DIR / "real_anomaly_windows.csv"
        anomalies.to_csv(out, index=False)
        n_ts = anomalies["timestamp"].nunique() if len(anomalies) > 0 else 0
        n_svc = anomalies["msname"].nunique() if len(anomalies) > 0 else 0
        print(f"  Found {len(anomalies)} anomalous readings "
              f"across {n_ts} timestamps, {n_svc} services")
        print(f"  Saved to {out}")

        # Cascades
        if len(anomalies) > 0:
            print("\nAnalyzing anomaly cascades...")
            cascades = find_anomaly_cascades(anomalies, cg_df)
            out = RESULTS_DIR / "anomaly_cascades.csv"
            cascades.to_csv(out, index=False)
            print(f"  Found {len(cascades)} cascade events")
            print(f"  Saved to {out}")

    elapsed = time.time() - total_t0
    print(f"\nAnalysis completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
