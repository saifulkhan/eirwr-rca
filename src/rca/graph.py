"""
Graph construction and matrix computation for RCA.

Implements:
  - Dependency graph extraction from CallGraph data
  - Weight matrix W (Eq. 1): normalized request rates
  - Fault Propagation Matrix P (Eq. 2): w_ij * (1 - R_i)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def build_dependency_graph(
    df: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    """
    Build the microservice dependency graph from CallGraph data.

    Extracts vertices V (unique services) and edges E (um -> dm pairs)
    with traffic volumes (request_count = lambda_ij) and average latency.

    Parameters
    ----------
    df : pd.DataFrame
        Raw CallGraph DataFrame with columns ``um``, ``dm``, ``rt``.

    Returns
    -------
    nodes : list[str]
        Sorted list of unique microservice names.
    edges_df : pd.DataFrame
        Aggregated edges with columns [um, dm, request_count, avg_latency].
    """
    clean = df.dropna(subset=["um", "dm", "rt"])
    # Filter out UNKNOWN placeholder nodes
    clean = clean[(clean["um"] != "UNKNOWN") & (clean["dm"] != "UNKNOWN")]

    edges_df = (
        clean.groupby(["um", "dm"])
        .agg(request_count=("rt", "count"), avg_latency=("rt", "mean"))
        .reset_index()
    )

    nodes = sorted(set(edges_df["um"]).union(edges_df["dm"]))
    return nodes, edges_df


def compute_weight_matrix(
    edges_df: pd.DataFrame,
    nodes: list[str],
) -> sparse.csr_matrix:
    """
    Compute the normalized dependency weight matrix W.

    Eq. 1: w_ij = lambda_ij / sum_k(lambda_ik)

    Each row i is normalized so outgoing edge weights sum to 1.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Aggregated edges with columns [um, dm, request_count].
    nodes : list[str]
        Ordered list of node names (defines matrix indices).

    Returns
    -------
    W : scipy.sparse.csr_matrix
        N x N sparse weight matrix.
    """
    node_to_idx: dict[str, int] = {name: idx for idx, name in enumerate(nodes)}
    n: int = len(nodes)

    rows, cols, vals = [], [], []
    for _, row in edges_df.iterrows():
        i = node_to_idx.get(row["um"])
        j = node_to_idx.get(row["dm"])
        if i is not None and j is not None:
            rows.append(i)
            cols.append(j)
            vals.append(row["request_count"])

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)

    # Normalize each row: w_ij = lambda_ij / sum_k(lambda_ik)
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # avoid division by zero for isolated nodes
    diag_inv = sparse.diags(1.0 / row_sums)
    W = diag_inv @ W

    return W


def compute_fault_propagation_matrix(
    W: sparse.csr_matrix,
    R: np.ndarray | float = 0.1,
) -> sparse.csr_matrix:
    """
    Compute the Fault Propagation Matrix P.

    Eq. 2: p_ji = w_ij * (1 - R_i)

    P[j, i] represents the probability that a failure in v_j cascades to v_i.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        N x N weight matrix.
    R : np.ndarray or float
        Resilience factor(s). Scalar applies uniformly; array is per-node.

    Returns
    -------
    P : scipy.sparse.csr_matrix
        N x N fault propagation matrix.
    """
    n = W.shape[0]
    if isinstance(R, (int, float)):
        R = np.full(n, R)

    # P[j, i] = W[i, j] * (1 - R[i])
    # So P = (diag(1 - R) @ W)^T  = W^T @ diag(1 - R)
    resilience_scale = sparse.diags(1.0 - R)
    P = (resilience_scale @ W).T

    return P.tocsr()
