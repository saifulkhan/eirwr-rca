"""
IRWR and eIRWR root cause localization.

- ``irwr``: anomaly-seeded power iteration on M_R = diag(1-R) W.
- ``irwr_enhanced`` (eIRWR): IRWR with anomaly-conditioned resilience,
  belief-weighted forward, backward and self transitions, and sharpened
  restart distribution.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def irwr(
    M_R: sparse.csr_matrix,
    s_obs: np.ndarray,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[np.ndarray, int]:
    """Iterative Random Walk with Restart (basic IRWR).

    r^(k+1) = (1 - alpha) * M_R @ r^(k) + alpha * s_obs,  r^(0) = s_obs

    Because W is row-stochastic over out-edges (caller -> callee), one
    application of M_R accumulates each callee's score into its callers.
    With uniform R this is equivalent to PPR with restart
    alpha_eff = 1 - (1 - alpha)(1 - R) (Proposition 1 in the paper).

    Parameters
    ----------
    M_R : scipy.sparse.csr_matrix
        Resilience-damped transition matrix diag(1-R) @ W (N x N).
        Equal to ``compute_fault_propagation_matrix(W, R).T``.
    s_obs : np.ndarray
        Observed anomaly vector (N,), normalized to sum to 1.
    alpha : float
        Restart probability.
    epsilon : float
        Convergence tolerance on the L1 norm.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    r : np.ndarray
        Root cause score vector (N,).
    k : int
        Number of iterations to converge.
    """
    r = s_obs.copy()
    for k in range(1, max_iter + 1):
        r_new = (1 - alpha) * M_R.dot(r) + alpha * s_obs
        error = np.abs(r_new - r).sum()
        r = r_new
        if error < epsilon:
            return r, k
    return r, max_iter


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
    adaptive_teleport: bool = True,
) -> tuple[np.ndarray, int]:
    """Enhanced IRWR (eIRWR), Algorithm 1 in the paper.

    Mechanisms (paper symbols in brackets):

    1. Anomaly-conditioned resilience: R_i = R_base * exp(-beta * s_i / max s).
    2. Belief-weighted forward transitions, plus backward edges with
       weight ``backward`` [rho] and self-loops for nodes whose belief
       exceeds that of every neighbour. Rows are re-normalized.
    3. Sharpened restart distribution v = b^q / ||b^q||_1 [q], where the
       exponent is reduced towards 1 when the belief peak-to-mean ratio
       is low (``adaptive_teleport``).
    4. Optional outer belief refinement with momentum [mu].

    Parameters
    ----------
    W : sparse.csr_matrix
        Row-stochastic weight matrix (caller -> callee).
    s_obs : np.ndarray
        Observed anomaly vector (N,), normalized to sum to 1.
    R : float
        Base resilience R_base.
    beta : float
        Resilience sensitivity to anomaly. 0 gives uniform resilience.
    alpha : float
        Restart probability.
    epsilon : float
        Inner convergence tolerance (L1 norm).
    max_iter : int
        Maximum inner iterations per outer step.
    n_outer : int
        Number of outer belief-refinement iterations.
    outer_epsilon : float
        Outer convergence tolerance.
    momentum : float
        Blend factor in [0, 1] between observation and current belief.
    backward : float
        Backward transition weight in [0, 1). 0 disables backward edges.
    teleport_q : float
        Exponent for restart sharpening. 1.0 disables sharpening.
    adaptive_teleport : bool
        Interpolate the exponent between 1 and ``teleport_q`` based on
        the belief peak-to-mean ratio (Eq. adaptive-q in the paper).

    Returns
    -------
    scores : np.ndarray
        Root cause score vector (N,).
    total_iters : int
        Total inner iterations across all outer steps.
    """
    # binary adjacency and its transpose (for backward edges)
    A = (W != 0).astype(np.float64)
    A_T = A.T.tocsr()

    # anomaly-conditioned resilience
    s_max = s_obs.max()
    s_norm = s_obs / s_max if s_max > 0 else s_obs.copy()
    R_adaptive = R * np.exp(-beta * s_norm)
    M_base = (sparse.diags(1.0 - R_adaptive) @ W).tocsr()

    r = s_obs.copy()
    total_iters = 0

    for outer in range(n_outer):
        belief = (1.0 - momentum) * s_obs + momentum * r
        b_max = belief.max()
        C = belief / b_max if b_max > 0 else belief.copy()

        # forward transitions weighted by the callee's belief
        M = M_base @ sparse.diags(C)

        # backward transitions where no forward edge exists
        if backward > 0:
            A_bwd = sparse.diags(backward * C) @ A_T
            A_bwd = A_bwd - A_bwd.multiply(A)
            A_bwd.eliminate_zeros()
            M = M + A_bwd

        # self-loops for nodes more suspicious than all their neighbours
        max_nb = np.asarray(M.max(axis=1).todense()).flatten()
        self_w = np.maximum(0.0, C - max_nb)
        M = M + sparse.diags(self_w)

        # row-normalize
        rs = np.asarray(M.sum(axis=1)).flatten()
        rs[rs == 0] = 1.0
        M = sparse.diags(1.0 / rs) @ M

        # sharpened restart distribution; weak signal (peak-to-mean < 5)
        # gives q = 1, strong signal (> 20) gives q = teleport_q
        q = teleport_q
        if adaptive_teleport:
            belief_mean = belief.mean()
            if belief_mean > 0:
                quality = belief.max() / belief_mean
                q = 1.0 + (teleport_q - 1.0) * min(
                    1.0, max(0.0, (quality - 5.0) / 15.0)
                )
            else:
                q = 1.0

        v = belief**q
        v_sum = v.sum()
        if v_sum > 0:
            v /= v_sum

        # inner power iteration, warm-started from the previous belief
        r_inner = r.copy()
        inner_k = 0
        for inner_k in range(1, max_iter + 1):  # noqa: B007
            r_next = (1 - alpha) * M.dot(r_inner) + alpha * v
            err = np.abs(r_next - r_inner).sum()
            r_inner = r_next
            if err < epsilon:
                break
        total_iters += inner_k

        outer_err = np.abs(r_inner - r).sum()
        r = r_inner
        if outer > 0 and outer_err < outer_epsilon:
            break

    return r, total_iters
