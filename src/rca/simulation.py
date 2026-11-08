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

    # root cause shows moderate anomaly (not dominant)
    s[root_cause_idx] = root_cause_visibility

    # propagate backward: callers of the faulty service observe symptoms.
    # W^T[i, j] = W[j, i] = weight of edge j -> i, so W^T propagates
    # influence from callees back to callers.
    W_T = W.T.tocsr()
    current = np.zeros(n)
    current[root_cause_idx] = 1.0

    for _step in range(propagation_steps):
        propagated = W_T.dot(current) * decay
        s += propagated
        current = propagated

    # add background noise to all nodes
    s += np.abs(rng.normal(0, noise_level, n))

    # normalize to create a proper probability distribution
    total = s.sum()
    if total > 0:
        s /= total

    return s
