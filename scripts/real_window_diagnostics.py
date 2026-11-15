"""Diagnostics for the real-window case study (paper Section 4.2).

Reproduces the numbers in the paragraph "Real-anomaly-window case study":
the number of services with a positive score per window, the largest
share of any service, the cascade coverage of the top-1 pick of every
method, its number of callers and its upstream cone, and the Spearman
correlation between coverage and cone size over all services.

Reads results/real_world_evaluation.csv (``evaluate_performance.py
--experiments real``) and MCRRTUpdate_<index>.csv. Writes
results/real_window_diagnostics.csv. Takes about five minutes on CG297.

usage: uv run python scripts/real_window_diagnostics.py [--dataset-index 297]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import load_msrt_mcr  # noqa: E402
from src.rca.real_eval import build_real_anomaly_vector  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
MAX_HOPS = 5
DECAY = 0.6


def load_prepare_graph():
    # reuse the graph construction of the evaluation script
    spec = importlib.util.spec_from_file_location(
        "evaluate_performance",
        PROJECT_ROOT / "scripts" / "evaluate_performance.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", type=int, default=297)
    args = parser.parse_args()

    t0 = time.time()
    graph = load_prepare_graph()(args.dataset_index)
    W, nodes = graph["W"].tocsr(), graph["nodes"]
    n = len(nodes)
    idx = {name: i for i, name in enumerate(nodes)}
    n_callers = np.diff(W.tocsc().indptr)
    WT = W.T.tocsr()

    res = pd.read_csv(RESULTS_DIR / "real_world_evaluation.csv")
    timestamps = sorted(res["timestamp"].unique())
    msr = load_msrt_mcr(args.dataset_index)
    S = np.stack(
        [build_real_anomaly_vector(msr, nodes, ts) for ts in timestamps]
    )

    # cascade coverage of every service, as in structural_plausibility_score:
    # weight 0.6^d for the service and its callers within five hops
    cov = np.zeros((len(timestamps), n))
    cone = np.zeros(n, dtype=int)
    for a in range(0, n, 1000):
        ids = np.arange(a, min(a + 1000, n))
        D = csgraph.dijkstra(
            WT, directed=True, indices=ids, unweighted=True, limit=MAX_HOPS
        )
        reach = np.isfinite(D)
        weight = np.where(reach, DECAY ** np.where(reach, D, 0), 0.0)
        cone[ids] = reach.sum(axis=1)
        cov[:, ids] = (weight @ S.T).T / S.sum(axis=1, keepdims=True)
    print(f"{n:,} services, coverage computed ({time.time() - t0:.0f}s)")

    rows = []
    for k, ts in enumerate(timestamps):
        s, c = S[k], cov[k]
        rho = spearmanr(c, cone).statistic
        print(
            f"\nwindow {ts}: {int(np.sum(s > 0))} positive services, "
            f"largest share {s.max() / s.sum():.4f}, "
            f"spearman(coverage, cone) {rho:.3f}"
        )
        for _, r in res[res["timestamp"] == ts].iterrows():
            j = idx[r["top_1"]]
            rows.append(
                {
                    "timestamp": ts,
                    "method": r["method"],
                    "top_1": r["top_1"],
                    "cascade_coverage": c[j],
                    "n_callers": int(n_callers[j]),
                    "upstream_cone": int(cone[j]),
                    "score_share": s[j] / s.sum(),
                    "n_positive_services": int(np.sum(s > 0)),
                    "max_score_share": s.max() / s.sum(),
                    "spearman_coverage_cone": rho,
                }
            )
            print(
                f"  {r['method']:15s} {r['top_1']:10s} coverage {c[j]:.4f}, "
                f"callers {n_callers[j]}, cone {cone[j]}, "
                f"score share {s[j] / s.sum():.4f}"
            )

    out = RESULTS_DIR / "real_window_diagnostics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
