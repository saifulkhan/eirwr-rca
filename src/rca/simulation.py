"""Anomaly simulation via chaos-engineering-style fault injection.

Simulates cascading failures from a root cause node through the
dependency graph with geometric decay and background noise.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def simulate_anomaly(
    W: sparse.csr_matrix,
    root_cause_idx: int,
    propagation_steps: int = 5,
    decay: float = 0.6,
    noise_level: float = 0.05,
    root_cause_visibility: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a cascading failure from a root cause node.

    Models a realistic monitoring scenario where:
    - The root cause service shows a moderate anomaly signal
    - Callers of the faulty service observe cascading latency/errors
    - Background noise exists across all services
    - The root cause is NOT necessarily the loudest signal

    Propagates anomaly scores *backward* through the dependency graph
    (from callee to callers) using W^T, with geometric decay at each hop.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        N x N weight matrix (caller -> callee direction).
    root_cause_idx : int
        Index of the injected root cause node.
    propagation_steps : int
        Number of hops to propagate the anomaly.
    decay : float
        Multiplicative decay per hop (0 < decay < 1).
    noise_level : float
        Std-dev of Gaussian noise added to all nodes.
    root_cause_visibility : float
        Anomaly score of the root cause itself (0-1). Lower values
        make the problem harder (root cause is less obviously anomalous).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    s_obs : np.ndarray
        Simulated observed anomaly vector (N,), normalized to sum to 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = W.shape[0]
    s = np.zeros(n)

    # Root cause shows moderate anomaly (not dominant)
    s[root_cause_idx] = root_cause_visibility

    # Propagate backward: callers of the faulty service observe symptoms.
    # W^T[i, j] = W[j, i] = weight of edge j -> i, so W^T propagates
    # influence from callees back to callers.
    W_T = W.T.tocsr()
    current = np.zeros(n)
    current[root_cause_idx] = 1.0

    for _step in range(propagation_steps):
        propagated = W_T.dot(current) * decay
        s += propagated
        current = propagated

    # Add background noise to all nodes
    s += np.abs(rng.normal(0, noise_level, n))

    # Normalize to create a proper probability distribution
    total = s.sum()
    if total > 0:
        s /= total

    return s


def simulate_anomaly_calibrated(
    W: sparse.csr_matrix,
    root_cause_idx: int,
    noise_cv: np.ndarray | None = None,
    anomaly_magnitude: float | None = None,
    propagation_steps: int = 5,
    decay: float = 0.6,
    root_cause_visibility: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a cascading failure with noise calibrated from real metrics.

    Instead of uniform Gaussian noise, uses per-service coefficient of
    variation (CV) from real MSRTMCR data to generate realistic background
    noise.  The anomaly magnitude at the root cause uses the real
    p99/p50 ratio to set a realistic signal strength.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        N x N weight matrix (caller -> callee direction).
    root_cause_idx : int
        Index of the injected root cause node.
    noise_cv : np.ndarray, optional
        Per-node coefficient of variation from real metrics.  Shape (N,).
        If None, falls back to uniform noise (noise_level=0.05).
    anomaly_magnitude : float, optional
        Anomaly strength at root cause (derived from real p99/p50 ratios).
        If None, uses default ``root_cause_visibility``.
    propagation_steps : int
        Number of hops to propagate the anomaly.
    decay : float
        Multiplicative decay per hop (0 < decay < 1).
    root_cause_visibility : float
        Anomaly score of the root cause itself (0-1).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    s_obs : np.ndarray
        Simulated observed anomaly vector (N,), normalized to sum to 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = W.shape[0]
    s = np.zeros(n)

    # Root cause anomaly (possibly calibrated)
    rc_vis = anomaly_magnitude if anomaly_magnitude is not None else root_cause_visibility
    s[root_cause_idx] = rc_vis

    # Propagate backward through callers
    W_T = W.T.tocsr()
    current = np.zeros(n)
    current[root_cause_idx] = 1.0

    for _step in range(propagation_steps):
        propagated = W_T.dot(current) * decay
        s += propagated
        current = propagated

    # Add per-service calibrated noise
    if noise_cv is not None:
        # Each node gets noise proportional to its real CV
        # Clip CV to reasonable range
        cv = np.clip(noise_cv[:n] if len(noise_cv) >= n else
                      np.pad(noise_cv, (0, n - len(noise_cv)), constant_values=0.05),
                      0.01, 0.5)
        noise = np.abs(rng.normal(0, cv))
    else:
        noise = np.abs(rng.normal(0, 0.05, n))

    s += noise

    # Normalize
    total = s.sum()
    if total > 0:
        s /= total

    return s
