"""Semi-supervised evaluation on real anomaly windows.

Builds anomaly vectors from real MSRTMCR data and scores RCA output
with label-free structural plausibility metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


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
    ranked: np.ndarray,
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    k: int = 5,
) -> dict[str, float]:
    """Compute plausibility metrics for an RCA result without ground truth.

    Metrics
    -------
    downstream_anomaly_ratio : float
        Fraction of top-k's callees that are also anomalous.
        High value = plausible (anomaly cascades outward from root cause).
    anomaly_centrality : float
        Average anomaly score of top-k predictions.
        Higher means the method identifies actually anomalous services.
    cascade_coverage : float
        Fraction of total anomaly explained by the cascade subgraph
        (all services reachable via outgoing calls) rooted at the
        top-1 prediction.
    """
    n = W.shape[0]
    top_k = ranked[:k]

    # 1. Downstream anomaly ratio
    anomaly_threshold = (
        np.percentile(s_obs[s_obs > 0], 50) if (s_obs > 0).any() else 0
    )
    n_anomalous_callees = 0
    n_total_callees = 0
    for node in top_k:
        if node >= n:
            continue
        for callee in W[node].indices:
            n_total_callees += 1
            if s_obs[callee] > anomaly_threshold:
                n_anomalous_callees += 1
    downstream_ratio = (
        n_anomalous_callees / n_total_callees if n_total_callees > 0 else 0.0
    )

    # 2. Anomaly centrality of predictions
    anomaly_centrality = float(np.mean(s_obs[top_k[top_k < n]]))

    # 3. Cascade coverage from top-1
    top1 = ranked[0]
    covered = np.zeros(n)
    if top1 < n:
        # BFS from top-1 through outgoing edges
        visited = set()
        queue = [int(top1)]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            covered[node] = 1.0
            for child in W[node].indices:
                if child not in visited:
                    queue.append(int(child))

    cascade_anomaly = float((s_obs * covered).sum())
    total_anomaly = float(s_obs.sum())
    cascade_coverage = (
        cascade_anomaly / total_anomaly if total_anomaly > 0 else 0.0
    )

    return {
        "downstream_anomaly_ratio": downstream_ratio,
        "anomaly_centrality": anomaly_centrality,
        "cascade_coverage": cascade_coverage,
    }
