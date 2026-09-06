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


def precision_at_k_value(ranked_indices: np.ndarray, true_idx: int, k: int) -> float:
    """Precision@K: Fraction of top-K predictions that are correct.

    For single root cause RCA:
    - Returns 1/K if true root cause is in top-K
    - Returns 0 otherwise

    Parameters
    ----------
    ranked_indices : np.ndarray
        Indices of nodes sorted by score (descending).
    true_idx : int
        Index of the true root cause.
    k : int
        Number of top predictions to consider.

    Returns
    -------
    precision : float
        Precision value in [0, 1/K].
    """
    if true_idx in ranked_indices[:k]:
        return 1.0 / k
    return 0.0


def recall_at_k(ranked_indices: np.ndarray, true_idx: int, k: int) -> float:
    """Recall@K: Fraction of true root causes found in top-K.

    For single root cause RCA:
    - Returns 1.0 if true root cause is in top-K
    - Returns 0.0 otherwise

    This is equivalent to precision_at_k (PR@K) for single root cause.

    Parameters
    ----------
    ranked_indices : np.ndarray
        Indices of nodes sorted by score (descending).
    true_idx : int
        Index of the true root cause.
    k : int
        Number of top predictions to consider.

    Returns
    -------
    recall : float
        Recall value in {0.0, 1.0}.
    """
    return float(true_idx in ranked_indices[:k])


def average_precision(ranked_indices: np.ndarray, true_idx: int) -> float:
    """Average Precision (AP): Area under precision-recall curve.

    For single root cause at rank r:
    AP = 1/r

    This is equivalent to reciprocal rank.

    Parameters
    ----------
    ranked_indices : np.ndarray
        Indices of nodes sorted by score (descending).
    true_idx : int
        Index of the true root cause.

    Returns
    -------
    ap : float
        Average precision in [0, 1].
    """
    rank = get_rank(ranked_indices, true_idx)
    return 1.0 / rank


def f1_score_at_k(ranked_indices: np.ndarray, true_idx: int, k: int) -> float:
    """F1 Score at K: Harmonic mean of precision and recall at K.

    F1 = 2 * (precision * recall) / (precision + recall)

    For single root cause RCA:
    - F1@K = 2/K if root cause in top-K
    - F1@K = 0 otherwise

    Parameters
    ----------
    ranked_indices : np.ndarray
        Indices of nodes sorted by score (descending).
    true_idx : int
        Index of the true root cause.
    k : int
        Number of top predictions to consider.

    Returns
    -------
    f1 : float
        F1 score at K.
    """
    prec = precision_at_k_value(ranked_indices, true_idx, k)
    rec = recall_at_k(ranked_indices, true_idx, k)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)
