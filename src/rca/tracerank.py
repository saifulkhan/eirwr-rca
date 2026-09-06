"""
TraceRank: Abnormal service localization with dis-aggregated
end-to-end tracing data in cloud native systems.

Strict implementation following Yu, Huang & Chen (2023),
J Softw Evol Proc, doi:10.1002/smr.2413

Components:
  - Spectrum scoring via Ochiai formula (Eq. 2)
  - Personalized PageRank with forward/backward/selfward transitions (Eq. 5-7)
  - Ranked List Correction Algorithm (Algorithm 2)
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy import sparse
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Spectrum scoring (Eq. 2)
# ---------------------------------------------------------------------------


def _ochiai_score(ef: int, ep: int, nf: int, np_: int) -> float:
    """Ochiai spectrum formula (Eq. 2).

    Score = ef / sqrt((ef + ep) * (ef + nf))
    """
    denom = math.sqrt((ef + ep) * (ef + nf))
    return ef / denom if denom > 0 else 0.0


def _compute_spectrum_scores(
    traces: list[dict[int, float]],
    is_anomalous: np.ndarray,
    n: int,
) -> np.ndarray:
    """Per-service Ochiai spectrum scores from disaggregated traces."""
    n_f = int(is_anomalous.sum())
    n_p = len(traces) - n_f

    ef = np.zeros(n, dtype=int)
    ep = np.zeros(n, dtype=int)
    for trace, anom in zip(traces, is_anomalous):
        for svc in trace:
            if svc < n:
                if anom:
                    ef[svc] += 1
                else:
                    ep[svc] += 1

    nf = n_f - ef
    np_arr = n_p - ep

    scores = np.zeros(n)
    for i in range(n):
        scores[i] = _ochiai_score(int(ef[i]), int(ep[i]), int(nf[i]), int(np_arr[i]))
    return scores


# ---------------------------------------------------------------------------
# Correlation scoring (Eq. 3)
# ---------------------------------------------------------------------------


def _compute_correlation_scores(
    traces: list[dict[int, float]],
    frontend_idx: int,
    n: int,
) -> np.ndarray:
    """PCC of each service's processing time with the frontend (Eq. 3).

    C_i = |PCC(processing_time_i, processing_time_frontend)|
    computed over all traces where both services appear.
    """
    svc_times: list[list[float]] = [[] for _ in range(n)]
    fe_times: list[list[float]] = [[] for _ in range(n)]

    for trace in traces:
        if frontend_idx not in trace:
            continue
        ft = trace[frontend_idx]
        for svc, pt in trace.items():
            if svc != frontend_idx and svc < n:
                svc_times[svc].append(pt)
                fe_times[svc].append(ft)

    C = np.zeros(n)
    C[frontend_idx] = 1.0

    for svc in range(n):
        if svc == frontend_idx or len(svc_times[svc]) < 3:
            continue
        x = np.array(svc_times[svc])
        y = np.array(fe_times[svc])
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        r, _ = pearsonr(x, y)
        C[svc] = abs(r) if not np.isnan(r) else 0.0

    return C


# ---------------------------------------------------------------------------
# Augmented adjacency matrix (Eq. 5)
# ---------------------------------------------------------------------------


def _build_augmented_matrix(
    W: sparse.csr_matrix,
    C: np.ndarray,
    rho: float,
    frontend_idx: int,
) -> sparse.csr_matrix:
    """Build augmented adjacency A' (Eq. 5).

    Forward:  A'[i,j] = C[j]           where edge i -> j exists
    Backward: A'[j,i] = rho * C[i]     where edge i -> j exists (no forward j -> i)
    Selfward: A'[i,i] = max(0, C[i] - max_{child k} C[k])
    Front-end exception: A'[frontend, frontend] = 0
    """
    A = (W != 0).astype(np.float64)
    A_T = A.T.tocsr()

    # Forward: weight by callee's correlation
    A_fwd = A @ sparse.diags(C)

    # Backward: A_bwd[j,i] = rho * C[i] where A[i,j] != 0
    A_bwd = A_T @ sparse.diags(rho * C)
    # Remove backward entries where a forward edge already exists
    A_bwd = A_bwd - A_bwd.multiply(A)
    A_bwd.eliminate_zeros()

    # Selfward
    max_child_C = np.asarray(A_fwd.max(axis=1).todense()).flatten()
    self_weights = np.maximum(0.0, C - max_child_C)
    self_weights[frontend_idx] = 0.0  # Front-end exception
    A_self = sparse.diags(self_weights)

    return (A_fwd + A_bwd + A_self).tocsr()


# ---------------------------------------------------------------------------
# Ranked List Correction (Algorithm 2)
# ---------------------------------------------------------------------------


def _ranked_list_correction(
    ppr_ranked: np.ndarray,
    spectrum_scores: np.ndarray,
    traces: list[dict[int, float]],
    is_anomalous: np.ndarray,
    n: int,
    gamma: float,
    k: int,
) -> np.ndarray:
    """Algorithm 2: Ranked List Correction.

    Promotes services that are both spectrally suspicious and have
    statistically abnormal latency in anomalous traces.
    """
    k = min(k, n)
    spectrum_topk = set(int(i) for i in np.argsort(-spectrum_scores)[:k])

    # Per-service latency statistics
    D_a_sum = np.zeros(n)
    D_a_count = np.zeros(n)
    D_n_vals: list[list[float]] = [[] for _ in range(n)]

    for trace, anom in zip(traces, is_anomalous):
        for svc, pt in trace.items():
            if svc >= n:
                continue
            if anom:
                D_a_sum[svc] += pt
                D_a_count[svc] += 1
            else:
                D_n_vals[svc].append(pt)

    D_a_count[D_a_count == 0] = 1
    D_a_mean = D_a_sum / D_a_count

    D_n_mean = np.zeros(n)
    D_n_std = np.zeros(n)
    for svc in range(n):
        if D_n_vals[svc]:
            D_n_mean[svc] = np.mean(D_n_vals[svc])
            D_n_std[svc] = np.std(D_n_vals[svc])

    # Build corrected ranking
    promoted: list[int] = []
    demoted: list[int] = []
    for svc in ppr_ranked:
        is_abnormal = D_a_mean[svc] > D_n_mean[svc] + gamma * D_n_std[svc]
        if is_abnormal and svc in spectrum_topk:
            promoted.append(svc)
        else:
            demoted.append(svc)

    final_order = promoted + demoted

    # Higher score = more suspicious
    scores = np.zeros(n)
    for rank, svc in enumerate(final_order):
        scores[svc] = float(n - rank)
    return scores


# ---------------------------------------------------------------------------
# Main TraceRank function
# ---------------------------------------------------------------------------


def tracerank(
    W: sparse.csr_matrix,
    traces: list[dict[int, float]],
    is_anomalous: np.ndarray,
    frontend_idx: int = 0,
    d: float = 0.15,
    rho: float = 0.3,
    gamma: float = 3.0,
    spectrum_k: int = 3,
    epsilon: float = 1e-6,
    max_iter: int = 1000,
) -> np.ndarray:
    """TraceRank root cause localization (Yu et al., 2023).

    Combines personalized PageRank with spectrum-based fault localization
    using disaggregated end-to-end tracing data.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        N x N row-stochastic weight matrix (caller -> callee).
    traces : list[dict[int, float]]
        Disaggregated traces.  Each trace maps service_idx -> processing_time.
    is_anomalous : np.ndarray of bool
        True for anomalous traces, False for normal.
    frontend_idx : int
        Index of the entry-point (front-end) service.
    d : float
        Damping factor.  Lower values = more frequent teleportation.
        Paper finds lower *d* works better.
    rho : float
        Backward transition discount in [0, 1).
    gamma : float
        Outlier threshold multiplier for Algorithm 2.
    spectrum_k : int
        Top-k from spectrum ranking used in Algorithm 2.
    epsilon : float
        Convergence tolerance (L1 norm).
    max_iter : int
        Maximum power-iteration steps.

    Returns
    -------
    scores : np.ndarray (N,)
        Root cause suspiciousness score per service.
    """
    n = W.shape[0]

    # Step 1: Spectrum scores (Eq. 2)
    spectrum = _compute_spectrum_scores(traces, is_anomalous, n)

    # Step 2: Correlation scores (Eq. 3)
    C = _compute_correlation_scores(traces, frontend_idx, n)

    # Step 3: Augmented adjacency matrix (Eq. 5)
    A_prime = _build_augmented_matrix(W, C, rho, frontend_idx)

    # Step 4: Transition probability matrix (Eq. 6)
    row_sums = np.asarray(A_prime.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    P = sparse.diags(1.0 / row_sums) @ A_prime

    # Step 5: Teleportation vector v (Eq. 7)
    v = C.copy()
    v[frontend_idx] = 0.0  # Front-end exception
    v_sum = v.sum()
    v = v / v_sum if v_sum > 0 else np.ones(n) / n

    # Step 6: Power iteration (Eq. 7)
    x = v.copy()
    for _ in range(max_iter):
        x_new = d * P.T.dot(x) + (1 - d) * v
        if np.abs(x_new - x).sum() < epsilon:
            x = x_new
            break
        x = x_new

    # Step 7: Ranked List Correction (Algorithm 2)
    ppr_ranked = np.argsort(-x)
    scores = _ranked_list_correction(
        ppr_ranked, spectrum, traces, is_anomalous, n, gamma, spectrum_k,
    )
    return scores


# ---------------------------------------------------------------------------
# Trace data simulation for evaluation
# ---------------------------------------------------------------------------


def simulate_trace_data(
    W: sparse.csr_matrix,
    root_cause_idx: int,
    n_anomalous: int = 50,
    n_normal: int = 150,
    base_latency: float = 10.0,
    anomaly_factor: float = 5.0,
    noise_std: float = 3.5,
    cascade_decay: float = 0.5,
    follow_prob: float = 0.7,
    root_cause_visibility: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[list[dict[int, float]], np.ndarray, int]:
    """Simulate disaggregated trace data for TraceRank evaluation.

    Generates traces by random walks through the dependency graph.
    Anomalous traces pass through the root cause with elevated latency
    that cascades to upstream callers.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        N x N weight matrix (caller -> callee).
    root_cause_idx : int
        Root cause service index.
    n_anomalous, n_normal : int
        Number of anomalous / normal traces.
    base_latency : float
        Base processing time per service.
    anomaly_factor : float
        Latency multiplier at root cause in anomalous traces.
    noise_std : float
        Processing time noise standard deviation. Higher values make
        anomaly detection harder (more realistic for low-visibility scenarios).
    cascade_decay : float
        Decay for cascading elevated latency to callers of root cause.
    follow_prob : float
        Probability of following each outgoing edge during walk.
    root_cause_visibility : float
        Controls how strongly the root cause and its cascade are elevated
        in anomalous traces (0.0 = normal latency, 1.0 = full anomaly).
        Both root cause latency and cascade to callers are scaled by
        this parameter to create comparable difficulty to ``simulate_anomaly``.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    traces : list[dict[int, float]]
        Each trace maps service_idx -> processing_time.
    is_anomalous : np.ndarray of bool
    frontend_idx : int
    """
    n = W.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    # -- Identify the frontend (entry-point) service -------------------------
    in_weights = np.asarray(W.sum(axis=0)).flatten()
    out_weights = np.asarray(W.sum(axis=1)).flatten()

    fe_candidates = np.where((out_weights > 0) & (in_weights < 1e-10))[0]
    if len(fe_candidates) > 0:
        frontend_idx = int(fe_candidates[0])
    else:
        ratio = out_weights / np.maximum(in_weights, 1e-10)
        frontend_idx = int(np.argmax(ratio))

    # -- Adjacency list ------------------------------------------------------
    W_csr = W.tocsr()
    adj: list[list[int]] = []
    for i in range(n):
        row = W_csr[i].toarray().flatten()
        adj.append(np.where(row > 0)[0].tolist())

    # -- BFS path from frontend to root cause --------------------------------
    def bfs_path(start: int, target: int) -> list[int] | None:
        queue: deque[tuple[int, list[int]]] = deque([(start, [start])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for child in adj[node]:
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        return None

    path_to_root = bfs_path(frontend_idx, root_cause_idx)

    # -- Cascade weights (from root cause through callers) -------------------
    W_T = W.T.tocsr()
    cascade: dict[int, float] = {}
    cq: deque[tuple[int, float]] = deque([(root_cause_idx, 1.0)])
    visited_c: set[int] = set()
    while cq:
        node, factor = cq.popleft()
        if node in visited_c:
            continue
        visited_c.add(node)
        cascade[node] = factor
        row = W_T[node].toarray().flatten()
        for caller in np.where(row > 0)[0]:
            if caller not in visited_c:
                cq.append((caller, factor * cascade_decay))

    # -- Trace generator -----------------------------------------------------
    def make_trace(force_root: bool) -> dict[int, float]:
        trace: dict[int, float] = {}
        stack = [frontend_idx]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            trace[node] = max(0.1, base_latency + rng.normal(0, noise_std))
            for child in adj[node]:
                if child not in seen and rng.random() < follow_prob:
                    stack.append(child)

        if force_root and root_cause_idx not in trace:
            if path_to_root is not None:
                for node in path_to_root:
                    if node not in trace:
                        trace[node] = max(
                            0.1, base_latency + rng.normal(0, noise_std)
                        )
            else:
                trace[root_cause_idx] = max(
                    0.1, base_latency + rng.normal(0, noise_std)
                )
        return trace

    # -- Generate anomalous traces -------------------------------------------
    traces: list[dict[int, float]] = []
    labels: list[bool] = []

    for _ in range(n_anomalous):
        trace = make_trace(force_root=True)
        # Root cause latency: scaled by visibility (0 = normal, 1 = full anomaly)
        root_elevation = root_cause_visibility * base_latency * (anomaly_factor - 1)
        trace[root_cause_idx] = max(
            0.1, base_latency + root_elevation + rng.normal(0, noise_std)
        )
        # Cascade to callers: scaled by visibility so difficulty matches s_obs
        for node, factor in cascade.items():
            if node in trace and node != root_cause_idx:
                trace[node] += (
                    base_latency * (anomaly_factor - 1)
                    * factor * root_cause_visibility
                )
        traces.append(trace)
        labels.append(True)

    # -- Generate normal traces ----------------------------------------------
    # Include root cause path in some normal traces so that spectrum scoring
    # cannot trivially discriminate (in real systems normal requests also
    # reach the root cause service).
    for _ in range(n_normal):
        include_rc = rng.random() < 0.5
        traces.append(make_trace(force_root=include_rc))
        labels.append(False)

    return traces, np.array(labels), frontend_idx
