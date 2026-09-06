"""Scan all CallGraph archives and select diverse topologies for evaluation.

Scans structural properties of all 500 CallGraph datasets and selects
a diverse subset using k-medoids clustering on graph features.

Usage:
    python scripts/select_topologies.py
    python scripts/select_topologies.py --n-select 5 --include 297
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import list_call_graph_archives, scan_call_graph_stats

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def scan_all_topologies(
    cache_path: Path | None = None,
    force_rescan: bool = False,
) -> pd.DataFrame:
    """Scan every archive and return structural summary.

    Results are cached to ``results/topology_stats.csv``.
    """
    if cache_path is None:
        cache_path = RESULTS_DIR / "topology_stats.csv"

    if cache_path.exists() and not force_rescan:
        print(f"Loading cached topology stats from {cache_path}")
        return pd.read_csv(cache_path)

    archives = list_call_graph_archives()
    print(f"Scanning {len(archives)} CallGraph archives...")

    rows = []
    for i, idx in enumerate(archives):
        t0 = time.time()
        try:
            stats = scan_call_graph_stats(idx)
            n = stats["n_nodes"]
            e = stats["n_unique_edges"]
            density = e / (n * (n - 1)) if n > 1 else 0.0
            avg_degree = e / n if n > 0 else 0.0

            rows.append({
                "index": idx,
                "n_rows": stats["n_rows"],
                "n_nodes": n,
                "n_edges": e,
                "density": density,
                "avg_degree": avg_degree,
                "file_size_mb": stats["file_size_mb"],
            })
            elapsed = time.time() - t0
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{len(archives)}] CG_{idx}: "
                      f"{n:,} nodes, {e:,} edges ({elapsed:.1f}s)")
        except Exception as exc:
            print(f"  [{i + 1}/{len(archives)}] CG_{idx}: FAILED - {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"\nSaved topology stats to {cache_path}")
    return df


def select_diverse_topologies(
    stats_df: pd.DataFrame,
    n_select: int = 5,
    include_indices: list[int] | None = None,
) -> list[int]:
    """Select topologies covering diverse structural regimes.

    Uses k-medoids clustering on (log(n_nodes), log(n_edges), density).
    Always includes specified indices for backward compatibility.
    """
    if include_indices is None:
        include_indices = [297]

    df = stats_df.copy()
    # Filter to valid graphs
    df = df[df["n_nodes"] > 10].reset_index(drop=True)

    # Feature matrix (log-scaled for better clustering)
    features = np.column_stack([
        np.log1p(df["n_nodes"].values),
        np.log1p(df["n_edges"].values),
        df["density"].values * 1000,  # scale density up
    ])

    # Normalize features
    mu = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    features_norm = (features - mu) / std

    # Simple k-medoids via greedy selection (maximise min distance to selected)
    selected_rows = []
    remaining = set(range(len(df)))

    # Force-include specified indices
    for idx in include_indices:
        mask = df["index"] == idx
        if mask.any():
            row_idx = int(df[mask].index[0])
            selected_rows.append(row_idx)
            remaining.discard(row_idx)

    # Greedily add most distant point from current selection
    while len(selected_rows) < n_select and remaining:
        best_row = -1
        best_min_dist = -1

        for candidate in remaining:
            min_dist = min(
                np.linalg.norm(features_norm[candidate] - features_norm[s])
                for s in selected_rows
            ) if selected_rows else float("inf")

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_row = candidate

        if best_row >= 0:
            selected_rows.append(best_row)
            remaining.discard(best_row)

    selected_indices = df.iloc[selected_rows]["index"].astype(int).tolist()

    # Print selection summary
    print(f"\nSelected {len(selected_indices)} diverse topologies:")
    print(f"{'Index':>6} {'Nodes':>8} {'Edges':>8} {'Density':>10} {'AvgDeg':>8}")
    print("-" * 48)
    for idx in selected_indices:
        row = df[df["index"] == idx].iloc[0]
        print(f"{idx:6d} {row['n_nodes']:8,} {row['n_edges']:8,} "
              f"{row['density']:10.6f} {row['avg_degree']:8.2f}")

    return selected_indices


def main():
    parser = argparse.ArgumentParser(description="Select diverse CallGraph topologies")
    parser.add_argument("--n-select", type=int, default=5, help="Number of topologies to select")
    parser.add_argument("--include", nargs="+", type=int, default=[297],
                        help="Indices to always include (default: [297])")
    parser.add_argument("--rescan", action="store_true", help="Force rescan all archives")
    args = parser.parse_args()

    stats_df = scan_all_topologies(force_rescan=args.rescan)
    selected = select_diverse_topologies(stats_df, args.n_select, args.include)

    # Save selection
    out = RESULTS_DIR / "selected_topologies.csv"
    sel_df = stats_df[stats_df["index"].isin(selected)]
    sel_df.to_csv(out, index=False)
    print(f"\nSaved selection to {out}")
    print(f"\nUse with: python scripts/evaluate_performance.py "
          f"--dataset-indices {' '.join(map(str, selected))}")


if __name__ == "__main__":
    main()
