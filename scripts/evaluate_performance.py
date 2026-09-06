"""
Comprehensive RCA Performance Evaluation Script.

Consolidates experiments from:
  - notebook/performance/performance.ipynb
  - notebook/performance/dataset-sensitivity.ipynb
  - notebook/performance/computational.ipynb

Evaluates proposed methods (IRWR, eIRWR) against baselines
across multiple dataset sizes and visibility levels. Saves all metrics
to CSV files in results/.

Usage:
    python scripts/evaluate_performance.py
    python scripts/evaluate_performance.py --n-incidents 50 --quick
    python scripts/evaluate_performance.py --dataset-index 297
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import load_call_graph_by_index
from src.rca.baselines import (
    coral_rca,
    degree_centrality,
    micro_rank,
    microhecl,
    monitor_rank,
    neighbor_correlation,
    personalized_pagerank,
    second_order_rw,
    tracediag,
)
from src.rca.graph import (
    build_dependency_graph,
    compute_fault_propagation_matrix,
    compute_weight_matrix,
)
from src.rca.irwr import irwr, irwr_enhanced
from src.rca.metrics import (
    average_precision,
    get_rank,
    mean_reciprocal_rank,
    precision_at_k,
    precision_at_k_value,
    recall_at_k,
)
from src.rca.simulation import simulate_anomaly
from src.rca.tracerank import simulate_trace_data, tracerank

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Helper functions ─────────────────────────────────────────────────────


def prepare_graph(dataset_index: int, nrows: int | None = None) -> dict:
    """Load dataset and build graph structures."""
    df = load_call_graph_by_index(dataset_index, nrows=nrows)
    nodes, edges_df = build_dependency_graph(df)
    W = compute_weight_matrix(edges_df, nodes)
    P = compute_fault_propagation_matrix(W, R=0.1)
    P_T = P.T.tocsr()
    callee_indices = sorted(
        {nodes.index(dm) for dm in edges_df["dm"].unique() if dm in nodes}
    )
    return {
        "df": df,
        "nodes": nodes,
        "edges_df": edges_df,
        "W": W,
        "P": P,
        "P_T": P_T,
        "callee_indices": callee_indices,
        "n_rows": len(df),
        "n_nodes": len(nodes),
        "n_edges": len(edges_df),
    }


def compute_all_metrics(ranked: np.ndarray, root_idx: int) -> dict[str, float]:
    """Compute full metric suite for a single incident."""
    rank = get_rank(ranked, root_idx)
    return {
        "rank": rank,
        "PR@1": precision_at_k(ranked, root_idx, 1),
        "PR@3": precision_at_k(ranked, root_idx, 3),
        "PR@5": precision_at_k(ranked, root_idx, 5),
        "Precision@1": precision_at_k_value(ranked, root_idx, 1),
        "Precision@3": precision_at_k_value(ranked, root_idx, 3),
        "Precision@5": precision_at_k_value(ranked, root_idx, 5),
        "Recall@1": recall_at_k(ranked, root_idx, 1),
        "Recall@3": recall_at_k(ranked, root_idx, 3),
        "Recall@5": recall_at_k(ranked, root_idx, 5),
        "AP": average_precision(ranked, root_idx),
    }


def aggregate_metrics(all_incident_metrics: list[dict]) -> dict[str, float]:
    """Aggregate per-incident metrics into summary statistics."""
    ranks = [m["rank"] for m in all_incident_metrics]
    return {
        "PR@1": np.mean([m["PR@1"] for m in all_incident_metrics]),
        "PR@3": np.mean([m["PR@3"] for m in all_incident_metrics]),
        "PR@5": np.mean([m["PR@5"] for m in all_incident_metrics]),
        "Precision@1": np.mean(
            [m["Precision@1"] for m in all_incident_metrics]
        ),
        "Precision@3": np.mean(
            [m["Precision@3"] for m in all_incident_metrics]
        ),
        "Precision@5": np.mean(
            [m["Precision@5"] for m in all_incident_metrics]
        ),
        "Recall@1": np.mean([m["Recall@1"] for m in all_incident_metrics]),
        "Recall@3": np.mean([m["Recall@3"] for m in all_incident_metrics]),
        "Recall@5": np.mean([m["Recall@5"] for m in all_incident_metrics]),
        "MRR": mean_reciprocal_rank(ranks),
        "MAP": np.mean([m["AP"] for m in all_incident_metrics]),
        "avg_rank": np.mean(ranks),
        "median_rank": np.median(ranks),
    }


def bench_time(fn, n_repeats: int = 10) -> tuple[float, float]:
    """Return (mean_seconds, std_seconds)."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def bench_memory(fn) -> int:
    """Return peak memory (bytes) during fn()."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


# ── Method runners ───────────────────────────────────────────────────────


def run_method(
    method_name: str,
    W,
    P_T,
    s_obs: np.ndarray,
    root_idx: int,
    visibility: float,
    seed: int,
    incident_idx: int,
) -> np.ndarray:
    """Run a single RCA method and return scores array."""
    if method_name == "MicroRCA":
        return personalized_pagerank(W, s_obs)
    elif method_name == "CloudRanger":
        # Use 50K walks for large graphs (>10K nodes), 10K for smaller graphs
        n_nodes = W.shape[0]
        n_walks = 50000 if n_nodes > 10000 else 10000
        return second_order_rw(
            W,
            s_obs,
            n_walks=n_walks,
            walk_length=20,
            rng=np.random.default_rng(seed * 10000 + incident_idx),
        )
    elif method_name == "In-Degree":
        return degree_centrality(W)
    elif method_name == "MicroHECL":
        return microhecl(W, s_obs)
    elif method_name == "TraceDiag":
        return tracediag(W, s_obs)
    elif method_name == "Neighbor Corr.":
        return neighbor_correlation(W, s_obs)
    elif method_name == "IRWR":
        r, _ = irwr(P_T, s_obs, alpha=0.15)
        return r
    elif method_name == "eIRWR":
        r, _ = irwr_enhanced(W, s_obs)
        return r
    elif method_name == "TraceRank":
        tr_rng = np.random.default_rng(seed * 10000 + incident_idx)
        traces, is_anom, fe_idx = simulate_trace_data(
            W,
            root_idx,
            root_cause_visibility=visibility,
            rng=tr_rng,
        )
        return tracerank(W, traces, is_anom, frontend_idx=fe_idx)
    elif method_name == "MonitorRank":
        return monitor_rank(W, s_obs)
    elif method_name == "MicroRank":
        tr_rng = np.random.default_rng(seed * 10000 + incident_idx)
        traces, is_anom, fe_idx = simulate_trace_data(
            W,
            root_idx,
            root_cause_visibility=visibility,
            rng=tr_rng,
        )
        return micro_rank(W, s_obs, traces, is_anom)
    elif method_name == "CORAL":
        return coral_rca(W, s_obs)
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ── Experiment 1: Scale × Visibility ────────────────────────────────────


def experiment_scale_visibility(
    dataset_index: int,
    nrows_list: list[int | None],
    visibilities: list[float],
    methods: list[str],
    n_incidents: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate all methods across dataset sizes and visibility levels."""
    print("=" * 70)
    print("EXPERIMENT 1: Scale × Visibility Performance")
    print("=" * 70)

    results = []

    for nrows in nrows_list:
        label = f"{nrows // 1000}K" if nrows else "FULL"
        print(f"\n{'─' * 60}")
        print(f"  Dataset: CallGraph_{dataset_index}, nrows={label}")
        t0 = time.time()
        graph = prepare_graph(dataset_index, nrows=nrows)
        print(
            f"  Graph: {graph['n_nodes']:,} nodes, {graph['n_edges']:,} edges"
            f" ({time.time() - t0:.1f}s)"
        )

        for vis in visibilities:
            rng = np.random.default_rng(seed)
            method_incidents: dict[str, list[dict]] = {m: [] for m in methods}

            t0 = time.time()
            for i_inc in range(n_incidents):
                root_idx = rng.choice(graph["callee_indices"])
                s_obs = simulate_anomaly(
                    graph["W"],
                    root_idx,
                    propagation_steps=5,
                    decay=0.6,
                    noise_level=0.05,
                    root_cause_visibility=vis,
                    rng=rng,
                )

                for method in methods:
                    scores = run_method(
                        method,
                        graph["W"],
                        graph["P_T"],
                        s_obs,
                        root_idx,
                        vis,
                        seed,
                        i_inc,
                    )
                    ranked = np.argsort(-scores)
                    metrics = compute_all_metrics(ranked, root_idx)
                    method_incidents[method].append(metrics)

            elapsed = time.time() - t0

            for method in methods:
                agg = aggregate_metrics(method_incidents[method])
                results.append(
                    {
                        "dataset_index": dataset_index,
                        "nrows": nrows or graph["n_rows"],
                        "n_nodes": graph["n_nodes"],
                        "n_edges": graph["n_edges"],
                        "visibility": vis,
                        "method": method,
                        "n_incidents": n_incidents,
                        **agg,
                    }
                )

            print(f"    vis={vis:.2f}: {elapsed:.1f}s")

    df = pd.DataFrame(results)
    out = RESULTS_DIR / "scale_visibility_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


# ── Experiment 2: Full comparison on large dataset ──────────────────────


def experiment_full_comparison(
    dataset_index: int,
    visibilities: list[float],
    methods: list[str],
    n_incidents: int,
    seed: int,
) -> pd.DataFrame:
    """Full comparison on the largest dataset (all rows)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Full Comparison on Large Dataset")
    print("=" * 70)

    t0 = time.time()
    graph = prepare_graph(dataset_index, nrows=None)
    print(
        f"  Full graph: {graph['n_nodes']:,} nodes, {graph['n_edges']:,} edges"
        f" ({time.time() - t0:.1f}s)"
    )

    results = []

    for vis in visibilities:
        rng = np.random.default_rng(seed)
        method_incidents: dict[str, list[dict]] = {m: [] for m in methods}

        t0 = time.time()
        for i_inc in range(n_incidents):
            root_idx = rng.choice(graph["callee_indices"])
            s_obs = simulate_anomaly(
                graph["W"],
                root_idx,
                propagation_steps=5,
                decay=0.6,
                noise_level=0.05,
                root_cause_visibility=vis,
                rng=rng,
            )

            for method in methods:
                scores = run_method(
                    method,
                    graph["W"],
                    graph["P_T"],
                    s_obs,
                    root_idx,
                    vis,
                    seed,
                    i_inc,
                )
                ranked = np.argsort(-scores)
                metrics = compute_all_metrics(ranked, root_idx)
                method_incidents[method].append(metrics)

        elapsed = time.time() - t0

        for method in methods:
            agg = aggregate_metrics(method_incidents[method])
            results.append(
                {
                    "dataset_index": dataset_index,
                    "n_rows": graph["n_rows"],
                    "n_nodes": graph["n_nodes"],
                    "n_edges": graph["n_edges"],
                    "visibility": vis,
                    "method": method,
                    "n_incidents": n_incidents,
                    **agg,
                }
            )

        print(f"  vis={vis:.2f}: {elapsed:.1f}s")

    df = pd.DataFrame(results)
    out = RESULTS_DIR / "full_comparison_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


# ── Experiment 3: Computational Benchmarks ──────────────────────────────


def experiment_computational(
    dataset_index: int,
    nrows_list: list[int | None],
    seed: int,
    n_time_repeats: int = 5,
) -> pd.DataFrame:
    """Measure wall-clock time and peak memory for each method."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Computational Benchmarks")
    print("=" * 70)

    records = []

    for nrows in nrows_list:
        label = f"{nrows // 1000}K" if nrows else "FULL"
        print(f"\n  nrows={label}...")
        graph = prepare_graph(dataset_index, nrows=nrows)
        W = graph["W"]
        P_T = graph["P_T"]
        n_nodes = graph["n_nodes"]
        n_edges = graph["n_edges"]

        rng = np.random.default_rng(seed)
        root_idx = rng.choice(graph["callee_indices"])
        s_obs = simulate_anomaly(
            W,
            root_idx,
            root_cause_visibility=0.3,
            rng=rng,
        )

        # Pre-generate trace data for TraceRank
        tr_rng = np.random.default_rng(seed)
        tr_traces, tr_is_anom, tr_fe_idx = simulate_trace_data(
            W,
            root_idx,
            rng=tr_rng,
        )

        # Determine n_walks based on graph size
        n_nodes = W.shape[0]
        cloudranger_walks = 50000 if n_nodes > 10000 else 10000

        bench_methods = {
            "MicroRCA": lambda: personalized_pagerank(W, s_obs),
            "CloudRanger": lambda: second_order_rw(
                W,
                s_obs,
                n_walks=cloudranger_walks,
                walk_length=20,
                rng=np.random.default_rng(0),
            ),
            "In-Degree": lambda: degree_centrality(W),
            "MicroHECL": lambda: microhecl(W, s_obs),
            "TraceDiag": lambda: tracediag(W, s_obs),
            "Neighbor Corr.": lambda: neighbor_correlation(W, s_obs),
            "IRWR": lambda: irwr(P_T, s_obs, alpha=0.15),
            "eIRWR": lambda: irwr_enhanced(W, s_obs),
            "TraceRank": lambda: tracerank(
                W,
                tr_traces,
                tr_is_anom,
                frontend_idx=tr_fe_idx,
            ),
            "MonitorRank": lambda: monitor_rank(W, s_obs),
            "MicroRank": lambda: micro_rank(W, s_obs, tr_traces, tr_is_anom),
            "CORAL": lambda: coral_rca(W, s_obs),
        }

        for method_name, fn in bench_methods.items():
            n_rep = 3 if method_name == "CloudRanger" else n_time_repeats
            mean_t, std_t = bench_time(fn, n_repeats=n_rep)
            peak_mem = bench_memory(fn)

            records.append(
                {
                    "dataset_index": dataset_index,
                    "nrows": nrows or graph["n_rows"],
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "method": method_name,
                    "mean_time_s": mean_t,
                    "std_time_s": std_t,
                    "peak_memory_bytes": peak_mem,
                    "peak_memory_mb": peak_mem / (1024 * 1024),
                }
            )
            print(
                f"    {method_name:20s}  "
                f"time={mean_t * 1000:8.2f}ms  "
                f"mem={peak_mem / 1024:8.1f}KB"
            )

    df = pd.DataFrame(records)
    out = RESULTS_DIR / "computational_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


# ── Experiment 4: Convergence Analysis ──────────────────────────────────


def experiment_convergence(
    dataset_index: int,
    nrows_list: list[int | None],
    alphas: list[float],
    seed: int,
    n_trials: int = 20,
) -> pd.DataFrame:
    """Measure convergence (iterations) vs alpha and graph size."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Convergence Analysis")
    print("=" * 70)

    records = []

    for nrows in nrows_list:
        label = f"{nrows // 1000}K" if nrows else "FULL"
        print(f"\n  nrows={label}...")
        graph = prepare_graph(dataset_index, nrows=nrows)
        W = graph["W"]

        for alpha in alphas:
            rng = np.random.default_rng(seed)
            irwr_iters = []
            eirwr_iters = []

            for _ in range(n_trials):
                root_idx = rng.choice(graph["callee_indices"])
                s_obs = simulate_anomaly(
                    W,
                    root_idx,
                    root_cause_visibility=0.3,
                    rng=rng,
                )

                P = compute_fault_propagation_matrix(W, R=0.1)
                P_T = P.T.tocsr()

                _, k1 = irwr(P_T, s_obs, alpha=alpha)
                irwr_iters.append(k1)

                _, k2 = irwr_enhanced(W, s_obs, alpha=alpha)
                eirwr_iters.append(k2)

            records.append(
                {
                    "dataset_index": dataset_index,
                    "nrows": nrows or graph["n_rows"],
                    "n_nodes": graph["n_nodes"],
                    "n_edges": graph["n_edges"],
                    "alpha": alpha,
                    "method": "IRWR",
                    "mean_iters": np.mean(irwr_iters),
                    "std_iters": np.std(irwr_iters),
                    "max_iters": np.max(irwr_iters),
                    "min_iters": np.min(irwr_iters),
                }
            )
            records.append(
                {
                    "dataset_index": dataset_index,
                    "nrows": nrows or graph["n_rows"],
                    "n_nodes": graph["n_nodes"],
                    "n_edges": graph["n_edges"],
                    "alpha": alpha,
                    "method": "eIRWR",
                    "mean_iters": np.mean(eirwr_iters),
                    "std_iters": np.std(eirwr_iters),
                    "max_iters": np.max(eirwr_iters),
                    "min_iters": np.min(eirwr_iters),
                }
            )

            print(
                f"    alpha={alpha:.2f}: "
                f"IRWR={np.mean(irwr_iters):.1f}iters, "
                f"eIRWR={np.mean(eirwr_iters):.1f}iters"
            )

    df = pd.DataFrame(records)
    out = RESULTS_DIR / "convergence_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


# ── Experiment 5: Per-Incident Detail (for statistical analysis) ────────


def experiment_per_incident(
    dataset_index: int,
    visibilities: list[float],
    methods: list[str],
    n_incidents: int,
    seed: int,
) -> pd.DataFrame:
    """Save per-incident ranks for statistical tests (Wilcoxon, etc.)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Per-Incident Ranks (for statistical tests)")
    print("=" * 70)

    t0 = time.time()
    graph = prepare_graph(dataset_index, nrows=None)
    print(
        f"  Full graph: {graph['n_nodes']:,} nodes, {graph['n_edges']:,} edges"
        f" ({time.time() - t0:.1f}s)"
    )

    records = []

    for vis in visibilities:
        rng = np.random.default_rng(seed)

        t0 = time.time()
        for i_inc in range(n_incidents):
            root_idx = rng.choice(graph["callee_indices"])
            s_obs = simulate_anomaly(
                graph["W"],
                root_idx,
                propagation_steps=5,
                decay=0.6,
                noise_level=0.05,
                root_cause_visibility=vis,
                rng=rng,
            )

            for method in methods:
                scores = run_method(
                    method,
                    graph["W"],
                    graph["P_T"],
                    s_obs,
                    root_idx,
                    vis,
                    seed,
                    i_inc,
                )
                ranked = np.argsort(-scores)
                metrics = compute_all_metrics(ranked, root_idx)
                records.append(
                    {
                        "incident_id": i_inc,
                        "visibility": vis,
                        "method": method,
                        "root_cause_idx": root_idx,
                        "root_cause_node": graph["nodes"][root_idx],
                        **metrics,
                    }
                )

        elapsed = time.time() - t0
        print(f"  vis={vis:.2f}: {elapsed:.1f}s")

    df = pd.DataFrame(records)
    out = RESULTS_DIR / "per_incident_ranks.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


# ── Experiment 7: Real-World Anomaly Evaluation ──────────────────────────


def experiment_real_world(
    dataset_index: int,
    methods: list[str],
    n_windows: int = 50,
    seed: int = 42,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Evaluate RCA methods on real anomaly windows from MSRTMCR data.

    Uses structural plausibility metrics since no ground truth exists.
    """
    from src.rca.real_eval import (
        build_real_anomaly_vector,
        structural_plausibility_score,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Real-World Anomaly Evaluation")
    print("=" * 70)

    # Load graph
    t0 = time.time()
    graph = prepare_graph(dataset_index, nrows=nrows)
    print(
        f"  Graph: {graph['n_nodes']:,} nodes, {graph['n_edges']:,} edges"
        f" ({time.time() - t0:.1f}s)"
    )

    # Load real anomaly windows
    anomaly_path = RESULTS_DIR / "real_anomaly_windows.csv"
    if not anomaly_path.exists():
        print(f"  ERROR: {anomaly_path} not found. "
              "Run scripts/analyze_real_metrics.py first.")
        return pd.DataFrame()

    anomalies = pd.read_csv(anomaly_path)
    anomaly_timestamps = sorted(anomalies["timestamp"].unique())
    if len(anomaly_timestamps) > n_windows:
        rng = np.random.default_rng(seed)
        anomaly_timestamps = rng.choice(
            anomaly_timestamps, size=n_windows, replace=False
        ).tolist()

    print(f"  Evaluating on {len(anomaly_timestamps)} real anomaly windows")

    # Load MSRTMCR for building s_obs
    msrtmcr_path = PROJECT_ROOT / "data" / "data" / "MSRTMCR" / "MCRRTUpdate_0.csv"
    if not msrtmcr_path.exists():
        print(f"  ERROR: {msrtmcr_path} not found.")
        return pd.DataFrame()

    from src.loader import load_msrt_mcr
    msrtmcr_df = load_msrt_mcr(path=msrtmcr_path)
    print(f"  Loaded MSRTMCR: {len(msrtmcr_df):,} rows")

    records = []
    for i, ts in enumerate(anomaly_timestamps):
        s_obs = build_real_anomaly_vector(
            msrtmcr_df, graph["nodes"], ts,
        )

        if s_obs.max() < 1e-6:
            continue

        for method in methods:
            scores = run_method(
                method, graph["W"], graph["P_T"], s_obs,
                root_idx=0, visibility=0.0, seed=seed, incident_idx=i,
            )
            ranked = np.argsort(-scores)
            plausibility = structural_plausibility_score(
                ranked, graph["W"], s_obs, k=5,
            )

            records.append({
                "timestamp": ts,
                "method": method,
                "top_1": graph["nodes"][ranked[0]],
                **plausibility,
            })

        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{len(anomaly_timestamps)}] windows processed")

    df = pd.DataFrame(records)
    out = RESULTS_DIR / "real_world_evaluation.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Print summary
    if len(df) > 0:
        summary = df.groupby("method").agg(
            avg_downstream=("downstream_anomaly_ratio", "mean"),
            avg_centrality=("anomaly_centrality", "mean"),
            avg_coverage=("cascade_coverage", "mean"),
        )
        print("\n  Method plausibility summary:")
        print(summary.to_string(float_format="%.4f"))

    return df


# ── Main ────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="RCA Performance Evaluation Script"
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=297,
        help="CallGraph file index (default: 297, the largest by nodes)",
    )
    parser.add_argument(
        "--n-incidents",
        type=int,
        default=200,
        help="Number of simulated incidents per configuration (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer sizes, visibilities, and incidents",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Which experiments to run (1-7, default: 1-5). "
             "6=statistical tests, 7=real-world evaluation",
    )
    parser.add_argument(
        "--dataset-indices",
        nargs="+",
        type=int,
        default=None,
        help="Multiple CallGraph indices for multi-topology evaluation. "
             "Overrides --dataset-index for experiments 2 and 5.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-computed (dataset_index, visibility, method) tuples",
    )
    return parser.parse_args()


def experiment_statistical_tests(
    per_incident_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Experiment 6: Statistical significance tests on per-incident data."""
    from src.rca.statistics import run_all_tests

    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Statistical Significance Tests")
    print("=" * 70)

    if per_incident_path is None:
        per_incident_path = RESULTS_DIR / "per_incident_ranks.csv"

    if not per_incident_path.exists():
        print(f"  ERROR: {per_incident_path} not found. Run experiment 5 first.")
        return {}

    df = pd.read_csv(per_incident_path)
    print(f"  Loaded {len(df):,} rows from {per_incident_path.name}")

    results = run_all_tests(df)

    for name, res_df in results.items():
        if len(res_df) > 0:
            out = RESULTS_DIR / f"statistical_{name}.csv"
            res_df.to_csv(out, index=False)
            print(f"  Saved: {out}")

    # Print summary
    if "friedman" in results and len(results["friedman"]) > 0:
        print("\n  Friedman test (omnibus):")
        for _, row in results["friedman"].iterrows():
            sig = "*" if row["significant"] else ""
            print(f"    vis={row['visibility']:.2f}: chi2={row['chi2']:.1f}, "
                  f"p={row['p_value']:.4g} {sig}")

    if "wilcoxon" in results and len(results["wilcoxon"]) > 0:
        print("\n  Wilcoxon signed-rank (significant at p<0.05):")
        sig_rows = results["wilcoxon"][results["wilcoxon"]["significant_005"]]
        for _, row in sig_rows.iterrows():
            print(f"    {row['proposed']} vs {row['baseline']} "
                  f"(vis={row['visibility']:.2f}): "
                  f"p={row['p_value']:.4g}, r={row['effect_size_r']:.3f}")

    return results


def _load_existing_results(path: Path) -> pd.DataFrame | None:
    """Load existing CSV results for resume support."""
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    args = parse_args()

    ALL_METHODS = [
        "MicroRCA",
        "CloudRanger",
        "In-Degree",
        "MicroHECL",
        "TraceDiag",
        "Neighbor Corr.",
        "MonitorRank",
        "MicroRank",
        "CORAL",
        "TraceRank",
        "IRWR",
        "eIRWR",
    ]

    # Multi-topology support
    dataset_indices = args.dataset_indices or [args.dataset_index]

    if args.quick:
        nrows_list = [500_000, 2_000_000, 5_000_000]
        visibilities = [0.0, 0.1, 0.3, 0.5]
        n_incidents = min(args.n_incidents, 50)
        comp_nrows = [500_000, 2_000_000]
        conv_nrows = [500_000, 2_000_000]
        alphas = [0.05, 0.15, 0.30]
    else:
        nrows_list = [
            100_000,
            500_000,
            1_000_000,
            2_000_000,
            5_000_000,
            10_000_000,
        ]
        visibilities = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        n_incidents = args.n_incidents
        comp_nrows = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
        conv_nrows = [500_000, 1_000_000, 2_000_000, 5_000_000]
        alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    print(f"Datasets: {['CallGraph_' + str(i) for i in dataset_indices]}")
    print(f"Incidents: {n_incidents}")
    print(f"Dataset sizes: {nrows_list}")
    print(f"Visibilities: {visibilities}")
    print(f"Methods: {ALL_METHODS}")
    print(f"Experiments: {args.experiments}")
    if args.resume:
        print("Resume mode: ON")
    print()

    total_t0 = time.time()

    # Experiments 1, 3, 4 run on primary dataset only
    primary_idx = dataset_indices[0]

    if 1 in args.experiments:
        experiment_scale_visibility(
            dataset_index=primary_idx,
            nrows_list=nrows_list,
            visibilities=visibilities,
            methods=ALL_METHODS,
            n_incidents=n_incidents,
            seed=args.seed,
        )

    # Experiments 2 and 5 run on all topologies
    if 2 in args.experiments:
        all_comparison_dfs = []
        for ds_idx in dataset_indices:
            checkpoint = RESULTS_DIR / f"full_comparison_metrics_CG{ds_idx}.csv"
            if args.resume and checkpoint.exists():
                print(f"\n  Resuming: loading {checkpoint.name}")
                all_comparison_dfs.append(pd.read_csv(checkpoint))
                continue
            result = experiment_full_comparison(
                dataset_index=ds_idx,
                visibilities=visibilities,
                methods=ALL_METHODS,
                n_incidents=n_incidents,
                seed=args.seed,
            )
            result.to_csv(checkpoint, index=False)
            all_comparison_dfs.append(result)

        if len(all_comparison_dfs) > 1:
            merged = pd.concat(all_comparison_dfs, ignore_index=True)
            out = RESULTS_DIR / "multi_topology_metrics.csv"
            merged.to_csv(out, index=False)
            print(f"\nSaved multi-topology results: {out}")

    if 3 in args.experiments:
        experiment_computational(
            dataset_index=primary_idx,
            nrows_list=comp_nrows,
            seed=args.seed,
        )

    if 4 in args.experiments:
        experiment_convergence(
            dataset_index=primary_idx,
            nrows_list=conv_nrows,
            alphas=alphas,
            seed=args.seed,
        )

    if 5 in args.experiments:
        all_incident_dfs = []
        for ds_idx in dataset_indices:
            checkpoint = RESULTS_DIR / f"per_incident_ranks_CG{ds_idx}.csv"
            if args.resume and checkpoint.exists():
                print(f"\n  Resuming: loading {checkpoint.name}")
                all_incident_dfs.append(pd.read_csv(checkpoint))
                continue
            result = experiment_per_incident(
                dataset_index=ds_idx,
                visibilities=visibilities,
                methods=ALL_METHODS,
                n_incidents=n_incidents,
                seed=args.seed,
            )
            result.to_csv(checkpoint, index=False)
            all_incident_dfs.append(result)

        if len(all_incident_dfs) > 1:
            merged = pd.concat(all_incident_dfs, ignore_index=True)
            out = RESULTS_DIR / "multi_topology_per_incident.csv"
            merged.to_csv(out, index=False)
            print(f"\nSaved multi-topology per-incident results: {out}")

    if 6 in args.experiments:
        experiment_statistical_tests()

    if 7 in args.experiments:
        experiment_real_world(
            dataset_index=primary_idx,
            methods=ALL_METHODS,
            seed=args.seed,
        )

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"All experiments completed in {total_elapsed:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
