"""Semi-supervised evaluation on real anomaly windows.

Builds anomaly vectors from real MSRTMCR data and scores RCA output
with label-free structural plausibility metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph


def build_real_anomaly_vector(
    msrtmcr_df: pd.DataFrame,
    nodes: list[str],
    timestamp: int,
    rt_col: str = "providerrpc_rt",
) -> np.ndarray:
    """Build s_obs from real metric data at a given timestamp.

    For each node in the dependency graph, computes anomaly score as:
        s_obs[i] = max(0, (RT_now[i] - RT_baseline[i]) / RT_baseline_std[i])

    where baseline is computed over the full time series.

    Parameters
    ----------
    msrtmcr_df : pd.DataFrame
        MSRTMCR data with columns [timestamp, msname, <rt_col>].
    nodes : list[str]
        Ordered list of service names from the dependency graph.
    timestamp : int
        Target timestamp for anomaly snapshot.
    rt_col : str
        Response time column to use.

    Returns
    -------
    s_obs : np.ndarray (N,) normalized to sum to 1.
    """
    n = len(nodes)
    node_to_idx = {name: i for i, name in enumerate(nodes)}

    valid = msrtmcr_df.dropna(subset=[rt_col])

    # compute per-service baselines
    baselines = valid.groupby("msname")[rt_col].agg(["mean", "std"])
    baselines.columns = ["base_mean", "base_std"]
    baselines["base_std"] = baselines["base_std"].replace(0, np.nan)

    # get values at target timestamp
    snapshot = valid[valid["timestamp"] == timestamp]
    # aggregate to per-service (mean across instances)
    svc_rt = snapshot.groupby("msname")[rt_col].mean()

    s_obs = np.zeros(n)
    for svc_name, rt_val in svc_rt.items():
        if svc_name not in node_to_idx:
            continue
        idx = node_to_idx[svc_name]
        if svc_name in baselines.index:
            base_mean = baselines.loc[svc_name, "base_mean"]
            base_std = baselines.loc[svc_name, "base_std"]
            if pd.notna(base_std) and base_std > 0:
                s_obs[idx] = max(0.0, (rt_val - base_mean) / base_std)

    # normalize
    total = s_obs.sum()
    if total > 0:
        s_obs /= total
    return s_obs


def structural_plausibility_score(
    top1: int,
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    max_hops: int = 5,
    decay: float = 0.6,
) -> dict[str, float]:
    """Label-free diagnostics of the top-1 service of an RCA result.

    Metrics
    -------
    downstream_anomaly_ratio : float
        Fraction of the callees of the top-1 service that are anomalous
        (score above the median of the positive scores). High for a
        victim whose callees are anomalous.
    cascade_coverage : float
        Share of the observed anomaly that a failure at the top-1
        service would explain under a simplified form of the fault model
        of the simulation (``simulate_anomaly``): a fault spreads from a
        callee to its callers, with ``decay`` per hop, for up to
        ``max_hops`` hops. Unlike the simulation, it uses the shortest
        hop distance and ignores the call shares.
        A caller at hop distance d has weight decay**d, the top-1
        service itself weight 1, and every other service weight 0.
    """
    positive = s_obs[s_obs > 0]
    threshold = np.median(positive) if len(positive) else 0.0
    callees = W[top1].indices
    downstream_ratio = (
        float(np.mean(s_obs[callees] > threshold)) if len(callees) else 0.0
    )

    # hop distance from each caller to the top-1 service, following call
    # edges backward (W^T), cut off after max_hops
    dist = csgraph.dijkstra(
        W.T, directed=True, indices=top1, unweighted=True, limit=max_hops
    )
    reached = np.isfinite(dist)
    weight = np.zeros(len(s_obs))
    weight[reached] = decay ** dist[reached]

    total = float(s_obs.sum())
    cascade_coverage = float(s_obs @ weight) / total if total > 0 else 0.0

    return {
        "downstream_anomaly_ratio": downstream_ratio,
        "cascade_coverage": cascade_coverage,
    }
