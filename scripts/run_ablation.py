"""Ablation study for the paper (Table: tab:ablation).

Decomposes the IRWR / eIRWR gain on one topology into the mechanisms
claimed by prior work and by us, evaluating every variant on the SAME
simulated incidents:

  1. MicroRCA          — PPR on W, default restart alpha=0.15
  2. PPR on W^T        — PPR on the true transpose (caller->callee walk)
  3. Basic IRWR        — power iteration on M_R = diag(1-R) W, R=0.1
  4. PPR tuned         — PPR on W with alpha_eff = 1-(1-alpha)(1-R) = 0.235
  5. eIRWR (full)     — enhanced method, code defaults

Rows 3 and 4 should produce identical rankings (Proposition 1 in the
paper: uniform resilience damping is an effective restart).

Usage:  uv run python scripts/run_ablation.py [--dataset-index 84]
        [--n-incidents 200] [--seed 42]
Writes: results/ablation_metrics.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_call_graph_by_index  # noqa: E402
from src.rca.baselines import personalized_pagerank  # noqa: E402
from src.rca.graph import (  # noqa: E402
    build_dependency_graph,
    compute_fault_propagation_matrix,
    compute_weight_matrix,
)
from src.rca.irwr import irwr, irwr_enhanced  # noqa: E402
from src.rca.metrics import (  # noqa: E402
    get_rank,
    mean_reciprocal_rank,
    precision_at_k,
)
from src.rca.simulation import simulate_anomaly  # noqa: E402

RESULTS_DIR = ROOT / "results"

ALPHA = 0.15
R_BASE = 0.1
ALPHA_EFF = 1 - (1 - ALPHA) * (1 - R_BASE)  # = 0.235


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", type=int, default=84)
    parser.add_argument("--n-incidents", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visibilities", type=float, nargs="+", default=[0.1, 0.3, 0.5]
    )
    args = parser.parse_args()

    print(f"Loading CallGraph_{args.dataset_index} (all rows)...")
    t0 = time.time()
    df = load_call_graph_by_index(args.dataset_index, nrows=None)
    nodes, edges_df = build_dependency_graph(df)
    W = compute_weight_matrix(edges_df, nodes)
    P = compute_fault_propagation_matrix(W, R=R_BASE)
    M_R = P.T.tocsr()  # = diag(1-R) W, the paper's M_R
    W_T = W.T.tocsr()
    callee_indices = sorted(
        {nodes.index(dm) for dm in edges_df["dm"].unique() if dm in nodes}
    )
    print(
        f"  {len(nodes):,} nodes, {len(edges_df):,} edges "
        f"({time.time() - t0:.1f}s)"
    )

    variants = {
        "MicroRCA (PPR on W)": lambda s: personalized_pagerank(
            W, s, alpha=ALPHA
        ),
        "PPR on W^T": lambda s: personalized_pagerank(W_T, s, alpha=ALPHA),
        "Basic IRWR (M_R)": lambda s: irwr(M_R, s, alpha=ALPHA)[0],
        f"PPR tuned (a={ALPHA_EFF:.3f})": lambda s: personalized_pagerank(
            W, s, alpha=ALPHA_EFF
        ),
        "eIRWR (full)": lambda s: irwr_enhanced(W, s)[0],
    }

    records = []
    for vis in args.visibilities:
        rng = np.random.default_rng(args.seed)
        per_variant: dict[str, list[dict]] = {v: [] for v in variants}

        t0 = time.time()
        for _ in range(args.n_incidents):
            root_idx = rng.choice(callee_indices)
            s_obs = simulate_anomaly(
                W,
                root_idx,
                propagation_steps=5,
                decay=0.6,
                noise_level=0.05,
                root_cause_visibility=vis,
                rng=rng,
            )
            for name, fn in variants.items():
                ranked = np.argsort(-fn(s_obs))
                per_variant[name].append(
                    {
                        "rank": get_rank(ranked, root_idx),
                        "PR@1": precision_at_k(ranked, root_idx, 1),
                        "PR@5": precision_at_k(ranked, root_idx, 5),
                    }
                )

        for name, metrics in per_variant.items():
            ranks = [m["rank"] for m in metrics]
            row = {
                "dataset_index": args.dataset_index,
                "visibility": vis,
                "variant": name,
                "n_incidents": args.n_incidents,
                "PR@1": float(np.mean([m["PR@1"] for m in metrics])),
                "PR@5": float(np.mean([m["PR@5"] for m in metrics])),
                "MRR": mean_reciprocal_rank(ranks),
                "avg_rank": float(np.mean(ranks)),
            }
            records.append(row)
            print(
                f"  v={vis:.1f}  {name:26s} "
                f"MRR={row['MRR']:.3f}  PR@1={row['PR@1']:.2f}  "
                f"PR@5={row['PR@5']:.2f}"
            )
        print(f"  ({time.time() - t0:.1f}s)")

    out = RESULTS_DIR / "ablation_metrics.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
