"""Evaluation metrics for Root Cause Analysis.

PR@K (Precision at K) and MRR (Mean Reciprocal Rank).
"""

from __future__ import annotations

import numpy as np


def precision_at_k(ranked_indices: np.ndarray, true_idx: int, k: int) -> int:
    """PR@K: 1 if the true root cause is in the top-k ranked nodes, else 0."""
    return int(true_idx in ranked_indices[:k])


def get_rank(ranked_indices: np.ndarray, true_idx: int) -> int:
    """Return the 1-based rank of true_idx in the ranked list."""
    positions = np.where(ranked_indices == true_idx)[0]
    if len(positions) == 0:
        return len(ranked_indices)
    return int(positions[0]) + 1


def mean_reciprocal_rank(ranks: list[int]) -> float:
    """MRR = 1/|I| * sum(1/rank_i) (Eq. 5)."""
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in ranks]))
