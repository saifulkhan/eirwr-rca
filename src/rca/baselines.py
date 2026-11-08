"""
Baseline RCA methods.

Aggregate-metric baselines (MicroRCA/PPR, CloudRanger, In-Degree,
MicroHECL, TraceDiag, Neighbour Correlation, MonitorRank, Corr-Fusion)
and the trace-level MicroRank baseline. Each is a re-implementation in
the spirit of the cited method, operating on the same weight matrix W.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from src.rca.tracerank import _compute_spectrum_scores

# ── 1: Personalized PageRank ─────────────────────────────────


def personalized_pagerank(
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    max_iter: int = 1000,
) -> np.ndarray:
    """
    Personalized PageRank (PPR) with anomaly teleport.

    Power iteration: r = (1-alpha) * W @ r + alpha * s_obs.
    This is the MicroRCA baseline (Wu et al., NOMS 2020). Basic IRWR
    runs the same iteration on M_R = diag(1-R) @ W instead of W.
    """
    r = s_obs.copy()

    for _ in range(max_iter):
        r_new = (1 - alpha) * W.dot(r) + alpha * s_obs
        if np.abs(r_new - r).sum() < epsilon:
            return r_new
        r = r_new
    return r


# ── 2: second-order Random Walk ──────────────────────────────


def second_order_rw(
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    n_walks: int = 50000,
    walk_length: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Second-order random walk (CloudRanger-style).

    Performs many random walks where transitions are biased by both
    the graph structure (edge weights) and the anomaly scores at
    neighboring nodes. The visit frequency of each node gives its
    root cause score. Inspired by CloudRanger (Wang et al., CNSM 2018).

    At each step, transition probability to neighbor j is proportional
    to W[current, j] * s_obs[j], favoring more anomalous neighbors.

    Parameters
    ----------
    n_walks : int, default=50000
        Number of random walks to perform. For large graphs (>10K nodes),
        use 50K-100K walks to ensure adequate coverage. For smaller graphs
        (<5K nodes), 10K walks may suffice.
    walk_length : int, default=20
        Maximum length of each random walk.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = W.shape[0]
    visit_count = np.zeros(n)

    W_csr = W.tocsr()

    start_probs = s_obs.copy()
    total = start_probs.sum()
    if total > 0:
        start_probs /= total
    else:
        start_probs = np.ones(n) / n

    for _ in range(n_walks):
        current = rng.choice(n, p=start_probs)
        visit_count[current] += 1

        for _ in range(walk_length):
            lo, hi = W_csr.indptr[current], W_csr.indptr[current + 1]
            neighbors = W_csr.indices[lo:hi]
            weights = W_csr.data[lo:hi]

            if len(neighbors) == 0:
                break

            anomaly_weights = weights * (s_obs[neighbors] + 1e-10)
            total_w = anomaly_weights.sum()
            if total_w == 0:
                break

            probs = anomaly_weights / total_w
            current = rng.choice(neighbors, p=probs)
            visit_count[current] += 1

    total = visit_count.sum()
    if total > 0:
        visit_count /= total
    return visit_count


# ── 3: in-Degree Centrality ─────────────────────────────────


def degree_centrality(W: sparse.csr_matrix) -> np.ndarray:
    """
    In-degree centrality ranking.

    Ranks nodes by their weighted in-degree (total incoming traffic).
    Nodes that receive more requests are ranked higher. This is the
    simplest structural baseline -- it assumes high-traffic services
    are more likely root causes.
    """
    in_degree = np.array(W.sum(axis=0)).flatten()
    return in_degree


# ── 4: MicroHECL ────────────────────────────────────────────


def microhecl(
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    max_depth: int = 5,
) -> np.ndarray:
    """
    MicroHECL-style forward graph traversal.

    Traces anomaly symptoms forward through the call graph (caller -> callee)
    to locate the deepest source of cascading failures. Starting from
    anomalous callers, propagates scores along edges to their callees,
    accumulating evidence at each hop. The callee that accumulates the
    most anomaly-weighted paths is the likely root cause.

    Inspired by MicroHECL (Alibaba, 2021): graph traversal + ranking
    along anomaly propagation chains in the call graph.
    """
    n = W.shape[0]
    scores = np.zeros(n)
    current = s_obs.copy()

    # forward propagation: caller -> callee (follow W directly)
    for _ in range(max_depth):
        propagated = W.dot(current)
        scores += propagated
        current = propagated

    return scores


# ── 5: TraceDiag ────────────────────────────────────────────


def tracediag(
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    prune_percentile: float = 50.0,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    max_iter: int = 1000,
) -> np.ndarray:
    """
    TraceDiag-style graph pruning + ranking.

    First prunes low-anomaly nodes (below a percentile threshold)
    to reduce noise, then runs Personalized PageRank on the pruned
    subgraph. Mapping scores back to the full node set.

    Inspired by TraceDiag (Microsoft, 2023): prune irrelevant nodes
    before causal analysis to improve scalability and accuracy.
    """
    n = W.shape[0]

    threshold = np.percentile(s_obs, prune_percentile)
    active = s_obs > threshold
    active_indices = np.where(active)[0]

    if len(active_indices) < 2:
        return s_obs.copy()

    W_sub = W[np.ix_(active_indices, active_indices)].tocsr()
    row_sums = np.array(W_sub.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W_sub = sparse.diags(1.0 / row_sums) @ W_sub

    s_sub = s_obs[active_indices]
    s_total = s_sub.sum()
    if s_total > 0:
        s_sub = s_sub / s_total

    r = s_sub.copy()
    for _ in range(max_iter):
        r_new = (1 - alpha) * W_sub.dot(r) + alpha * s_sub
        if np.abs(r_new - r).sum() < epsilon:
            r = r_new
            break
        r = r_new

    scores = np.zeros(n)
    scores[active_indices] = r
    return scores


# ── 6: neighbor Correlation ─────────────────────────────────


def neighbor_correlation(
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
) -> np.ndarray:
    """
    Neighbor anomaly correlation (PAL/FChain-style).

    Ranks nodes by how well their neighborhood anomaly pattern
    matches a root cause signature: a true root cause tends to have
    anomalous callers (showing cascade symptoms) regardless of its
    own anomaly level. Combines:
    - Upstream anomaly: aggregate anomaly of callers (cascade signal)
    - Own anomaly: the node's direct anomaly score
    - Downstream anomaly: aggregate anomaly of callees (secondary)

    Inspired by PAL and FChain: statistical correlation of KPI
    anomalies between connected services for fault localization.
    """
    caller_anomaly = W.T.dot(s_obs)
    callee_anomaly = W.dot(s_obs)
    scores = caller_anomaly + s_obs + 0.5 * callee_anomaly
    return scores


# ── 7: MonitorRank ─────────────────────────────────────────


def monitor_rank(
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    max_iter: int = 1000,
) -> np.ndarray:
    """MonitorRank: anomaly-correlated random walk.

    Modulates transition probabilities by anomaly correlation between
    connected services.  Edge (i, j) weight becomes
    W[i,j] * sqrt(s_obs[i]) * sqrt(s_obs[j]), keeping the matrix sparse.

    Reference: Kim et al., "Root Cause Detection in a Service-Oriented
    Architecture by Analyzing Dependency on Graph", 2013.
    """
    n = W.shape[0]

    # sparse anomaly-correlated weight matrix
    sqrt_s = np.sqrt(np.maximum(s_obs, 0))
    W_corr = sparse.diags(sqrt_s) @ W @ sparse.diags(sqrt_s)

    # row-normalize
    row_sums = np.asarray(W_corr.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    M = sparse.diags(1.0 / row_sums) @ W_corr

    # PPR with anomaly teleport
    v = s_obs.copy()
    v_sum = v.sum()
    v = v / v_sum if v_sum > 0 else np.ones(n) / n

    r = v.copy()
    for _ in range(max_iter):
        r_new = (1 - alpha) * M.dot(r) + alpha * v
        if np.abs(r_new - r).sum() < epsilon:
            return r_new
        r = r_new
    return r


# ── 8: MicroRank ──────────────────────────────────────────


def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1]."""
    xmin, xmax = x.min(), x.max()
    return (
        (x - xmin) / (xmax - xmin)
        if (xmax - xmin) > 1e-12
        else np.zeros_like(x)
    )


def micro_rank(
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    traces: list[dict[int, float]] | None = None,
    is_anomalous: np.ndarray | None = None,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    max_iter: int = 1000,
) -> np.ndarray:
    """MicroRank: Extended PageRank with spectrum analysis.

    Combines structural PageRank with spectrum-based fault localization
    (SBFL).  If trace data is provided, uses Ochiai scoring; otherwise
    falls back to an s_obs-based spectrum approximation.

    Reference: Yu et al., "MicroRank: End-to-End Latency Issue
    Localization with Extended Spectrum Analysis", IEEE TSC, 2021.
    """
    n = W.shape[0]

    # phase 1: Spectrum scoring
    if traces is not None and is_anomalous is not None:
        spectrum = _compute_spectrum_scores(traces, is_anomalous, n)
    else:
        # approximate spectrum from anomaly vector
        threshold = np.median(s_obs)
        ef = np.where(s_obs > threshold, s_obs, 0.0)
        ep_count = np.where(s_obs <= threshold, 1.0, 0.0)
        denom = np.sqrt((ef + ep_count) * (ef + ef.sum()))
        denom[denom == 0] = 1.0
        spectrum = ef / denom

    # phase 2: PageRank with spectrum-weighted teleportation
    v = spectrum.copy()
    v_sum = v.sum()
    v = v / v_sum if v_sum > 0 else np.ones(n) / n

    r = v.copy()
    for _ in range(max_iter):
        r_new = (1 - alpha) * W.T.dot(r) + alpha * v
        if np.abs(r_new - r).sum() < epsilon:
            break
        r = r_new

    # phase 3: Combine spectrum and PageRank scores
    return 0.5 * _normalize_minmax(spectrum) + 0.5 * _normalize_minmax(r)


# ── 9: Corr-Fusion (CORAL) ──────────────────────────────────────────────


def coral_rca(
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    beta: float = 0.5,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    max_iter: int = 1000,
) -> np.ndarray:
    """Corr-Fusion: correlation-fusion composite baseline.

    Builds a correlation-weighted adjacency by scaling edge weights
    with the anomaly scores of both endpoints, then blends it with the
    original dependency graph for random-walk RCA (in the spirit of
    MonitorRank-style correlation weighting).  Kept under the internal
    name "CORAL" in result CSVs; displayed as "Corr-Fusion" in the
    paper.
    """
    n = W.shape[0]

    # anomaly-correlation scaling (stays sparse)
    s_norm = s_obs / s_obs.max() if s_obs.max() > 0 else s_obs.copy()
    C = sparse.diags(s_norm) @ W @ sparse.diags(s_norm)

    # combined graph: beta * dependency + (1-beta) * correlation
    M = beta * W + (1 - beta) * C

    # row-normalize
    row_sums = np.asarray(M.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    M = sparse.diags(1.0 / row_sums) @ M

    # random walk with anomaly teleport
    v = s_obs.copy()
    v_sum = v.sum()
    v = v / v_sum if v_sum > 0 else np.ones(n) / n

    r = v.copy()
    for _ in range(max_iter):
        r_new = (1 - alpha) * M.dot(r) + alpha * v
        if np.abs(r_new - r).sum() < epsilon:
            return r_new
        r = r_new
    return r
