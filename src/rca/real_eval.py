"""Semi-supervised evaluation on real anomaly windows.

Provides functions to build anomaly vectors from real MSRTMCR data
and evaluate RCA methods using structural plausibility metrics
when ground truth is unavailable.
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

    # Compute per-service baselines
    baselines = valid.groupby("msname")[rt_col].agg(["mean", "std"])
    baselines.columns = ["base_mean", "base_std"]
    baselines["base_std"] = baselines["base_std"].replace(0, np.nan)

    # Get values at target timestamp
    snapshot = valid[valid["timestamp"] == timestamp]
    # Aggregate to per-service (mean across instances)
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

    # Normalize
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
        rooted at the top-1 prediction.
    top_k_agreement : float
        Average pairwise Jaccard similarity of top-k across methods
        (computed externally; placeholder here returns 0).
    """
    n = W.shape[0]
    top_k = ranked[:k]

    # 1. Downstream anomaly ratio
    anomaly_threshold = np.percentile(s_obs[s_obs > 0], 50) if (s_obs > 0).any() else 0
    n_anomalous_callees = 0
    n_total_callees = 0
    for node in top_k:
        if node >= n:
            continue
        row = W[node]
        callees = row.indices if hasattr(row, "indices") else np.where(row.toarray().flatten() > 0)[0]
        for callee in callees:
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
            row = W[node]
            children = row.indices if hasattr(row, "indices") else np.where(row.toarray().flatten() > 0)[0]
            for child in children:
                if child not in visited:
                    queue.append(int(child))

    cascade_anomaly = float((s_obs * covered).sum())
    total_anomaly = float(s_obs.sum())
    cascade_coverage = cascade_anomaly / total_anomaly if total_anomaly > 0 else 0.0

    return {
        "downstream_anomaly_ratio": downstream_ratio,
        "anomaly_centrality": anomaly_centrality,
        "cascade_coverage": cascade_coverage,
    }


def evaluate_on_real_incidents(
    W: sparse.csr_matrix,
    P_T: sparse.csr_matrix,
    nodes: list[str],
    anomaly_timestamps: list[int],
    msrtmcr_df: pd.DataFrame,
    method_runners: dict[str, callable],
    rt_col: str = "providerrpc_rt",
    k: int = 5,
) -> pd.DataFrame:
    """Run all RCA methods on real anomaly windows.

    Since ground truth is unavailable, evaluates using structural
    plausibility metrics.

    Parameters
    ----------
    W : sparse.csr_matrix
        Weight matrix.
    P_T : sparse.csr_matrix
        Fault propagation matrix transpose.
    nodes : list[str]
        Node names.
    anomaly_timestamps : list[int]
        Timestamps with detected anomalies.
    msrtmcr_df : pd.DataFrame
        Full MSRTMCR data for building s_obs.
    method_runners : dict[str, callable]
        Maps method name -> function(W, P_T, s_obs) -> scores.
    rt_col : str
        Response time column.
    k : int
        Top-k for plausibility evaluation.

    Returns
    -------
    pd.DataFrame with columns [timestamp, method, top_1, top_k,
                                downstream_anomaly_ratio,
                                anomaly_centrality, cascade_coverage].
    """
    records = []

    for ts in anomaly_timestamps:
        s_obs = build_real_anomaly_vector(msrtmcr_df, nodes, ts, rt_col)

        # Skip timestamps with negligible anomaly signal
        if s_obs.max() < 1e-6:
            continue

        for method_name, runner in method_runners.items():
            scores = runner(W, P_T, s_obs)
            ranked = np.argsort(-scores)

            plausibility = structural_plausibility_score(ranked, W, s_obs, k)

            records.append({
                "timestamp": ts,
                "method": method_name,
                "top_1": nodes[ranked[0]] if ranked[0] < len(nodes) else "?",
                "top_k": [nodes[r] for r in ranked[:k] if r < len(nodes)],
                **plausibility,
            })

    return pd.DataFrame(records)


def rank_stability(
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    runner: callable,
    n_perturbations: int = 20,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> float:
    """Measure rank stability under small perturbations to s_obs.

    Returns the fraction of perturbations where the top-1 prediction
    remains unchanged.  Higher = more confident diagnosis.
    """
    if rng is None:
        rng = np.random.default_rng()

    base_scores = runner(W, None, s_obs)
    base_top1 = np.argmax(base_scores)

    stable = 0
    for _ in range(n_perturbations):
        perturbed = s_obs + np.abs(rng.normal(0, noise_std, len(s_obs)))
        total = perturbed.sum()
        if total > 0:
            perturbed /= total
        scores = runner(W, None, perturbed)
        if np.argmax(scores) == base_top1:
            stable += 1

    return stable / n_perturbations
