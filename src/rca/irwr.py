from __future__ import annotations

import numpy as np
from scipy import sparse


def irwr(
    P_T: sparse.csr_matrix,
    s_obs: np.ndarray,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[np.ndarray, int]:
    """
    Iterative Random Walk with Restart

    Power iteration on P^T to trace observed anomaly symptoms backward
    through the dependency graph and identify the most likely root cause.

    r^(k+1) = (1 - alpha) * P^T @ r^(k) + alpha * s_obs

    Parameters
    ----------
    P_T : scipy.sparse.csr_matrix
        Transpose of the fault propagation matrix (N x N).
    s_obs : np.ndarray
        Observed anomaly symptom vector (N,). Should be normalized.
    alpha : float
        Restart probability (default 0.15).
    epsilon : float
        Convergence tolerance on L1 norm (default 1e-6).
    max_iter : int
        Maximum iterations before stopping.

    Returns
    -------
    r : np.ndarray
        Root cause probability vector (N,).
    k : int
        Number of iterations to converge.
    """

    r = s_obs.copy()

    for k in range(1, max_iter + 1):
        r_new = (1 - alpha) * P_T.dot(r) + alpha * s_obs
        error = np.abs(r_new - r).sum()
        r = r_new
        if error < epsilon:
            return r, k

    return r, max_iter


def _ochiai_scores(s_obs: np.ndarray, threshold_quantile: float = 0.5) -> np.ndarray:
    """Compute Ochiai spectrum scores from anomaly observations.

    Nodes above the threshold quantile are treated as "anomalous" evidence (ef),
    nodes below as "normal" evidence (ep). Returns per-node suspiciousness.
    """
    threshold = np.quantile(s_obs, threshold_quantile)
    ef = np.where(s_obs > threshold, s_obs, 0.0)
    ep = np.where(s_obs <= threshold, 1.0 - s_obs, 0.0)
    denom = np.sqrt((ef + ep) * (ef + ef.sum()))
    denom[denom == 0] = 1.0
    return ef / denom


def _laplacian_denoise(
    W: sparse.csr_matrix, s_obs: np.ndarray, lam: float = 1.0
) -> np.ndarray:
    """Graph Laplacian denoising of the anomaly vector.

    Solves  (I + λ·L) · s_smooth = s_obs  where L = D − A_sym is the
    symmetrised graph Laplacian.  Smooths noise while preserving cascade
    patterns that respect dependency topology.
    """
    from scipy.sparse.linalg import spsolve

    n = W.shape[0]
    A_sym = (W + W.T).tocsr()
    A_sym.data[:] = np.minimum(A_sym.data, 1.0)
    d = np.asarray(A_sym.sum(axis=1)).flatten()
    L = sparse.diags(d) - A_sym
    s_smooth = spsolve((sparse.eye(n) + lam * L).tocsc(), s_obs)
    s_smooth = np.maximum(s_smooth, 0.0)
    s_sum = s_smooth.sum()
    return s_smooth / s_sum if s_sum > 0 else s_obs.copy()


def _nag_teleport(W: sparse.csr_matrix, s: np.ndarray) -> np.ndarray:
    """Neighbourhood Anomaly Gradient (NAG) teleportation vector.

    NAG[i] = max(0,  s[i] − Σ_j W[i,j]·s[j])

    Positive values mark *anomaly sinks*: nodes that are anomalous but
    whose downstream callees are not — the hallmark of a root cause.
    The frontier score is weighted by local anomaly magnitude to
    suppress false positives in low-anomaly regions.
    """
    nag = np.maximum(s - W.dot(s), 0.0)
    frontier = nag * s
    f_sum = frontier.sum()
    return frontier / f_sum if f_sum > 0 else s.copy()


def irwr_enhanced(
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    R: float = 0.1,
    beta: float = 2.0,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    max_iter: int = 500,
    n_outer: int = 1,
    outer_epsilon: float = 1e-4,
    momentum: float = 0.1,
    backward: float = 0.3,
    teleport_q: float = 2.0,
    spectrum: bool = False,
    adaptive_teleport: bool = True,
    denoise: float = 0.0,
    nag: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Fault-Propagation-Aware RWR (FP-RWR).

    Extends classical RWR with innovations grounded in the
    fault-propagation model P = diag(1-R) @ W.

    Performance characteristics:
    - Low visibility (0.0-0.1): Limited performance due to reliance on
      aggregate anomaly vector. TraceRank significantly outperforms by
      leveraging per-trace data.
    - Moderate visibility (0.2-0.3): Competitive performance, approaches
      TraceRank as signal strength increases.
    - High visibility (≥0.5): Strong performance (PR@1 > 0.9), though
      TraceRank may still achieve perfect scores.

    Key innovations:

    1. **Fault-propagation-weighted transitions**
       Uses the weighted dependency matrix W (call frequencies)
       instead of binary adjacency.  High-traffic edges propagate
       faults more strongly, matching real microservice dynamics.
       TraceRank discards call frequency by binarising edges.

    2. **Anomaly-conditioned resilience**
       R_i = R_base * exp(-beta * s_norm_i).  Degraded services
       have reduced resilience, propagating faults more readily.
       This is grounded in the physical model and has no
       counterpart in TraceRank.

    3. **Power-law teleportation sharpening**
       v = s_obs^q / ||s_obs^q||_1.  Concentrates restarts at the
       most anomalous nodes and suppresses background noise.
       Particularly effective at low root-cause visibility, where
       noise constitutes a larger fraction of the signal.

    An optional outer loop with conservative momentum (default 0.1)
    refines the transition weights using the evolving belief state.

    Parameters
    ----------
    W : sparse.csr_matrix
        Row-stochastic weight matrix (caller -> callee).
    s_obs : np.ndarray
        Observed anomaly vector (N,), normalized to sum to 1.
    R : float
        Base resilience factor.
    beta : float
        Resilience sensitivity to anomaly.  Higher values make
        anomalous nodes less resilient.  0 = uniform resilience.
    alpha : float
        RWR restart probability.
    epsilon : float
        Inner convergence tolerance (L1 norm).
    max_iter : int
        Maximum inner iterations per outer step.
    n_outer : int
        Outer belief-refinement iterations.  1 = single-pass.
    outer_epsilon : float
        Outer convergence tolerance.
    momentum : float
        Blend factor in [0, 1] for belief refinement.
        0 = static (no refinement).  Keep low (0.1) to avoid
        confirmation bias.
    backward : float
        Backward transition weight in [0, 1).  0 disables.
    teleport_q : float
        Power-law exponent for teleportation sharpening.
        1.0 = standard (no sharpening).  2.0 = recommended.
    spectrum : bool
        Apply Ochiai SBFL calibration after convergence. Default False.
        Note: This uses a pseudo-spectrum based on quantile thresholding
        of s_obs, not true trace-based spectrum like TraceRank. In practice,
        this calibration often hurts performance and is disabled by default.
    adaptive_teleport : bool
        Automatically adjust teleport_q based on signal quality.
        When signal is weak (hidden root cause), reduces q to 1.0
        for broader exploration. When signal is strong, uses the
        specified q value. Default True.
    denoise : float
        Graph Laplacian denoising strength (λ).  0 = disabled.
        Recommended range 0.5–2.0.  Pre-processes s_obs to suppress
        measurement noise while preserving cascade structure.
    nag : float
        NAG (Neighbourhood Anomaly Gradient) blend weight in [0, 1].
        0 = disabled, 1 = pure NAG teleportation.  Intermediate values
        blend NAG with power-law sharpening.  Recommended 0.2–0.4.

    Returns
    -------
    scores : np.ndarray
        Root-cause probability vector (N,).
    total_iters : int
        Total inner iterations across all outer steps.
    """
    # -- Graph Laplacian denoising -----------------------------------------
    if denoise > 0:
        s_obs = _laplacian_denoise(W, s_obs, lam=denoise)

    # Pre-compute binary adjacency and its transpose
    A = (W != 0).astype(np.float64)
    A_T = A.T.tocsr()

    # -- Anomaly-conditioned resilience ---------------------------------
    s_max = s_obs.max()
    s_norm = s_obs / s_max if s_max > 0 else s_obs.copy()
    R_adaptive = R * np.exp(-beta * s_norm)

    # Base forward matrix with adaptive resilience
    M_base = (sparse.diags(1.0 - R_adaptive) @ W).tocsr()

    # Initial belief = observation
    r = s_obs.copy()
    total_iters = 0

    for outer in range(n_outer):
        # -- Belief-aware correlation vector ----------------------------
        belief = (1.0 - momentum) * s_obs + momentum * r
        b_max = belief.max()
        C = belief / b_max if b_max > 0 else belief.copy()

        # -- Forward transitions weighted by belief ---------------------
        M = M_base @ sparse.diags(C)

        # -- Backward transitions ---------------------------------------
        if backward > 0:
            A_bwd = sparse.diags(backward * C) @ A_T
            A_bwd = A_bwd - A_bwd.multiply(A)
            A_bwd.eliminate_zeros()
            M = M + A_bwd

        # -- Self-loops for suspicious nodes ----------------------------
        max_nb = np.asarray(
            M.max(axis=1).todense()
        ).flatten()
        self_w = np.maximum(0.0, C - max_nb)
        M = M + sparse.diags(self_w)

        # -- Row-normalize to maintain stochasticity --------------------
        rs = np.asarray(M.sum(axis=1)).flatten()
        rs[rs == 0] = 1.0
        M = sparse.diags(1.0 / rs) @ M

        # -- Power-law sharpened teleportation --------------------------
        # Adaptive teleportation: use signal quality to decide sharpening
        if adaptive_teleport:
            # Signal quality metric: ratio of max to mean (Gini-like)
            belief_max = belief.max()
            belief_mean = belief.mean()
            if belief_mean > 0:
                signal_quality = belief_max / belief_mean
                # If signal is weak (quality < 5), use no sharpening (q=1)
                # If signal is strong (quality > 20), use full sharpening
                # Linear interpolation in between
                q_adaptive = 1.0 + (teleport_q - 1.0) * min(
                    1.0, max(0.0, (signal_quality - 5.0) / 15.0)
                )
            else:
                q_adaptive = 1.0
        else:
            q_adaptive = teleport_q

        v = belief ** q_adaptive
        v_sum = v.sum()
        if v_sum > 0:
            v /= v_sum
        if nag > 0:
            v_nag = _nag_teleport(W, belief)
            v = (1.0 - nag) * v + nag * v_nag

        # -- Inner power iteration (warm-start from previous belief) ----
        r_inner = r.copy()
        inner_k = 0
        for inner_k in range(1, max_iter + 1):
            r_next = (1 - alpha) * M.dot(r_inner) + alpha * v
            err = np.abs(r_next - r_inner).sum()
            r_inner = r_next
            if err < epsilon:
                break
        total_iters += inner_k

        # -- Outer convergence check ------------------------------------
        outer_err = np.abs(r_inner - r).sum()
        r = r_inner
        if outer > 0 and outer_err < outer_epsilon:
            break

    # -- Spectrum calibration (post-hoc) --------------------------------
    if spectrum:
        spec = _ochiai_scores(s_obs)
        r = r * (1.0 + spec)

    return r, total_iters


def irwr_multiresolution(
    W: sparse.csr_matrix,
    s_obs: np.ndarray,
    alphas: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25),
    **kwargs,
) -> tuple[np.ndarray, int]:
    """Multi-resolution eIRWR: fuse walks at multiple restart rates.

    Different alpha values capture different propagation distances:
    low alpha = long walks (deep root causes), high alpha = short walks
    (shallow root causes).  Scores from each resolution are max-normalised
    then averaged.

    Parameters
    ----------
    W, s_obs : as for ``irwr_enhanced``.
    alphas : tuple of float
        Restart probabilities to run.
    **kwargs
        Forwarded to ``irwr_enhanced`` (e.g. ``denoise=0.1``).

    Returns
    -------
    scores : np.ndarray
        Fused root-cause probability vector (N,).
    total_iters : int
        Sum of iterations across all resolutions.
    """
    all_scores = []
    total_iters = 0
    for alpha in alphas:
        r, k = irwr_enhanced(W, s_obs, alpha=alpha, **kwargs)
        total_iters += k
        r_max = r.max()
        all_scores.append(r / r_max if r_max > 0 else r)
    return np.mean(all_scores, axis=0), total_iters
