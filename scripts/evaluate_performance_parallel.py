"""
Parallelized RCA Performance Evaluation Script.

Runs experiments using multiprocessing to leverage multiple CPUs.
Uses joblib for better progress tracking and memory management.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_performance import (
    RESULTS_DIR,
    aggregate_metrics,
    bench_memory,
    bench_time,
    compute_all_metrics,
    prepare_graph,
    run_method,
)
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
from src.rca.graph import compute_fault_propagation_matrix
from src.rca.irwr import irwr, irwr_enhanced
from src.rca.simulation import simulate_anomaly
from src.rca.tracerank import simulate_trace_data, tracerank


# ── Parallelized Experiment 1: Scale × Visibility ──────────────────────


def _run_single_incident_all_methods(
    methods: list[str],
    W,
    P_T,
    root_idx: int,
    vis: float,
    seed: int,
    incident_idx: int,
) -> dict[str, dict]:
    """Run all methods for a single incident."""
    # Create fresh RNG from combined seed for reproducibility
    incident_seed = seed * 100000 + incident_idx
    rng = np.random.default_rng(incident_seed)
    s_obs = simulate_anomaly(
        W,
        root_idx,
        propagation_steps=5,
        decay=0.6,
        noise_level=0.05,
        root_cause_visibility=vis,
        rng=rng,
    )

    results = {}
    for method in methods:
        scores = run_method(
            method, W, P_T, s_obs, root_idx, vis, seed, incident_idx
        )
        ranked = np.argsort(-scores)
        metrics = compute_all_metrics(ranked, root_idx)
        results[method] = metrics

    return results


def _run_visibility_level(
    vis: float,
    graph: dict,
    methods: list[str],
    n_incidents: int,
    seed: int,
    n_jobs: int,
) -> list[dict]:
    """Run all incidents for a single visibility level in parallel."""
    rng = np.random.default_rng(seed)

    # Pre-generate all root causes
    tasks = []
    for i_inc in range(n_incidents):
        root_idx = rng.choice(graph["callee_indices"])
        tasks.append((root_idx, i_inc))

    # Run incidents in parallel
    incident_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_single_incident_all_methods)(
            methods, graph["W"], graph["P_T"], root_idx, vis, seed, i_inc
        )
        for root_idx, i_inc in tasks
    )

    # Aggregate results by method
    method_incidents = {m: [] for m in methods}
    for inc_result in incident_results:
        for method, metrics in inc_result.items():
            method_incidents[method].append(metrics)

    # Build result records
    records = []
    for method in methods:
        agg = aggregate_metrics(method_incidents[method])
        records.append(
            {
                "visibility": vis,
                "method": method,
                "n_incidents": n_incidents,
                **agg,
            }
        )
    return records


def experiment_scale_visibility_parallel(
    dataset_index: int,
    nrows_list: list[int | None],
    visibilities: list[float],
    methods: list[str],
    n_incidents: int,
    seed: int,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Evaluate all methods across dataset sizes and visibility levels (parallelized)."""
    print("=" * 70)
    print("EXPERIMENT 1: Scale × Visibility Performance (Parallelized)")
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

        # Run all visibility levels
        for vis in visibilities:
            t0 = time.time()
            vis_results = _run_visibility_level(
                vis, graph, methods, n_incidents, seed, n_jobs
            )
            elapsed = time.time() - t0

            # Add graph metadata
            for rec in vis_results:
                rec.update(
                    {
                        "dataset_index": dataset_index,
                        "nrows": nrows or graph["n_rows"],
                        "n_nodes": graph["n_nodes"],
                        "n_edges": graph["n_edges"],
                    }
                )
            results.extend(vis_results)
            print(f"    vis={vis:.2f}: {elapsed:.1f}s")

    df = pd.DataFrame(results)
    out = RESULTS_DIR / "scale_visibility_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


# ── Parallelized Experiment 2: Full Comparison ──────────────────────


def experiment_full_comparison_parallel(
    dataset_index: int,
    visibilities: list[float],
    methods: list[str],
    n_incidents: int,
    seed: int,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Full comparison on the largest dataset (parallelized)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Full Comparison on Large Dataset (Parallelized)")
    print("=" * 70)

    t0 = time.time()
    graph = prepare_graph(dataset_index, nrows=None)
    print(
        f"  Full graph: {graph['n_nodes']:,} nodes, {graph['n_edges']:,} edges"
        f" ({time.time() - t0:.1f}s)"
    )

    results = []
    for vis in visibilities:
        t0 = time.time()
        vis_results = _run_visibility_level(
            vis, graph, methods, n_incidents, seed, n_jobs
        )
        elapsed = time.time() - t0

        # Add graph metadata
        for rec in vis_results:
            rec.update(
                {
                    "dataset_index": dataset_index,
                    "n_rows": graph["n_rows"],
                    "n_nodes": graph["n_nodes"],
                    "n_edges": graph["n_edges"],
                }
            )
        results.extend(vis_results)
        print(f"  vis={vis:.2f}: {elapsed:.1f}s")

    df = pd.DataFrame(results)
    out = RESULTS_DIR / "full_comparison_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


# ── Parallelized Experiment 3: Computational Benchmarks ──────────────


def _bench_single_method(
    method_name: str,
    bench_fn,
    n_time_repeats: int,
) -> dict:
    """Benchmark a single method."""
    mean_t, std_t = bench_time(bench_fn, n_repeats=n_time_repeats)
    peak_mem = bench_memory(bench_fn)
    return {
        "method": method_name,
        "mean_time_s": mean_t,
        "std_time_s": std_t,
        "peak_memory_bytes": peak_mem,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }


def experiment_computational_parallel(
    dataset_index: int,
    nrows_list: list[int | None],
    seed: int,
    n_time_repeats: int = 5,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Measure wall-clock time and peak memory for each method (parallelized)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Computational Benchmarks (Parallelized)")
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
        s_obs = simulate_anomaly(W, root_idx, root_cause_visibility=0.3, rng=rng)

        # Pre-generate trace data
        tr_rng = np.random.default_rng(seed)
        tr_traces, tr_is_anom, tr_fe_idx = simulate_trace_data(W, root_idx, rng=tr_rng)

        cloudranger_walks = 50000 if n_nodes > 10000 else 10000

        # Build benchmark functions
        bench_methods = {
            "MicroRCA": lambda: personalized_pagerank(W, s_obs),
            "CloudRanger": lambda: second_order_rw(
                W, s_obs, n_walks=cloudranger_walks, walk_length=20,
                rng=np.random.default_rng(0)
            ),
            "In-Degree": lambda: degree_centrality(W),
            "MicroHECL": lambda: microhecl(W, s_obs),
            "TraceDiag": lambda: tracediag(W, s_obs),
            "Neighbor Corr.": lambda: neighbor_correlation(W, s_obs),
            "IRWR": lambda: irwr(P_T, s_obs, alpha=0.15),
            "eIRWR": lambda: irwr_enhanced(W, s_obs),
            "TraceRank": lambda: tracerank(W, tr_traces, tr_is_anom, frontend_idx=tr_fe_idx),
            "MonitorRank": lambda: monitor_rank(W, s_obs),
            "MicroRank": lambda: micro_rank(W, s_obs, tr_traces, tr_is_anom),
            "CORAL": lambda: coral_rca(W, s_obs),
        }

        # Run benchmarks in parallel
        method_results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_bench_single_method)(
                name,
                fn,
                3 if name == "CloudRanger" else n_time_repeats,
            )
            for name, fn in bench_methods.items()
        )

        # Add graph metadata
        for res in method_results:
            res.update(
                {
                    "dataset_index": dataset_index,
                    "nrows": nrows or graph["n_rows"],
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                }
            )
            records.append(res)
            print(
                f"    {res['method']:20s}  "
                f"time={res['mean_time_s'] * 1000:8.2f}ms  "
                f"mem={res['peak_memory_bytes'] / 1024:8.1f}KB"
            )

    df = pd.DataFrame(records)
    out = RESULTS_DIR / "computational_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


# ── Parallelized Experiment 4: Convergence Analysis ──────────────────


def _run_convergence_trial(
    W, P_T, root_idx: int, s_obs, alpha: float
) -> tuple[int, int]:
    """Single convergence trial for both IRWR and eIRWR."""
    _, k1 = irwr(P_T, s_obs, alpha=alpha)
    _, k2 = irwr_enhanced(W, s_obs, alpha=alpha)
    return k1, k2


def experiment_convergence_parallel(
    dataset_index: int,
    nrows_list: list[int | None],
    alphas: list[float],
    seed: int,
    n_trials: int = 20,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Measure convergence (iterations) vs alpha and graph size (parallelized)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Convergence Analysis (Parallelized)")
    print("=" * 70)

    records = []

    for nrows in nrows_list:
        label = f"{nrows // 1000}K" if nrows else "FULL"
        print(f"\n  nrows={label}...")
        graph = prepare_graph(dataset_index, nrows=nrows)
        W = graph["W"]

        for alpha in alphas:
            rng = np.random.default_rng(seed)

            # Pre-generate all trials
            tasks = []
            for _ in range(n_trials):
                root_idx = rng.choice(graph["callee_indices"])
                s_obs = simulate_anomaly(W, root_idx, root_cause_visibility=0.3, rng=rng)
                P = compute_fault_propagation_matrix(W, R=0.1)
                P_T = P.T.tocsr()
                tasks.append((W, P_T, root_idx, s_obs, alpha))

            # Run trials in parallel
            results = Parallel(n_jobs=n_jobs)(
                delayed(_run_convergence_trial)(W, P_T, root_idx, s_obs, alpha)
                for W, P_T, root_idx, s_obs, alpha in tasks
            )

            irwr_iters = [r[0] for r in results]
            eirwr_iters = [r[1] for r in results]

            for method, iters in [("IRWR", irwr_iters), ("eIRWR", eirwr_iters)]:
                records.append(
                    {
                        "dataset_index": dataset_index,
                        "nrows": nrows or graph["n_rows"],
                        "n_nodes": graph["n_nodes"],
                        "n_edges": graph["n_edges"],
                        "alpha": alpha,
                        "method": method,
                        "mean_iters": np.mean(iters),
                        "std_iters": np.std(iters),
                        "max_iters": np.max(iters),
                        "min_iters": np.min(iters),
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


# ── Parallelized Experiment 5: Per-Incident Detail ──────────────────


def _run_single_incident_detailed(
    methods: list[str],
    W,
    P_T,
    nodes: list,
    root_idx: int,
    vis: float,
    seed: int,
    incident_idx: int,
) -> list[dict]:
    """Run all methods for a single incident and return detailed records."""
    # Create fresh RNG from combined seed for reproducibility
    incident_seed = seed * 100000 + incident_idx
    rng = np.random.default_rng(incident_seed)
    s_obs = simulate_anomaly(
        W,
        root_idx,
        propagation_steps=5,
        decay=0.6,
        noise_level=0.05,
        root_cause_visibility=vis,
        rng=rng,
    )

    records = []
    for method in methods:
        scores = run_method(method, W, P_T, s_obs, root_idx, vis, seed, incident_idx)
        ranked = np.argsort(-scores)
        metrics = compute_all_metrics(ranked, root_idx)
        records.append(
            {
                "incident_id": incident_idx,
                "visibility": vis,
                "method": method,
                "root_cause_idx": root_idx,
                "root_cause_node": nodes[root_idx],
                **metrics,
            }
        )
    return records


def experiment_per_incident_parallel(
    dataset_index: int,
    visibilities: list[float],
    methods: list[str],
    n_incidents: int,
    seed: int,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Save per-incident ranks for statistical tests (parallelized)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Per-Incident Ranks (Parallelized)")
    print("=" * 70)

    t0 = time.time()
    graph = prepare_graph(dataset_index, nrows=None)
    print(
        f"  Full graph: {graph['n_nodes']:,} nodes, {graph['n_edges']:,} edges"
        f" ({time.time() - t0:.1f}s)"
    )

    all_records = []

    for vis in visibilities:
        rng = np.random.default_rng(seed)

        # Pre-generate tasks
        tasks = []
        for i_inc in range(n_incidents):
            root_idx = rng.choice(graph["callee_indices"])
            tasks.append((root_idx, i_inc))

        t0 = time.time()
        # Run in parallel
        incident_records = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_run_single_incident_detailed)(
                methods,
                graph["W"],
                graph["P_T"],
                graph["nodes"],
                root_idx,
                vis,
                seed,
                i_inc,
            )
            for root_idx, i_inc in tasks
        )

        # Flatten results
        for records_list in incident_records:
            all_records.extend(records_list)

        elapsed = time.time() - t0
        print(f"  vis={vis:.2f}: {elapsed:.1f}s")

    df = pd.DataFrame(all_records)
    out = RESULTS_DIR / "per_incident_ranks.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


# ── Main ────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallelized RCA Performance Evaluation Script"
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=297,
        help="CallGraph file index (default: 297)",
    )
    parser.add_argument(
        "--n-incidents",
        type=int,
        default=200,
        help="Number of simulated incidents per configuration (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
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
        help="Which experiments to run (1-5, default: 1-5)",
    )
    parser.add_argument(
        "--dataset-indices",
        nargs="+",
        type=int,
        default=None,
        help="Multiple CallGraph indices for multi-topology evaluation",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 = all CPUs, default: -1)",
    )
    return parser.parse_args()


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

    dataset_indices = args.dataset_indices or [args.dataset_index]

    if args.quick:
        nrows_list = [500_000, 2_000_000, 5_000_000]
        visibilities = [0.0, 0.1, 0.3, 0.5]
        n_incidents = min(args.n_incidents, 50)
        comp_nrows = [500_000, 2_000_000]
        conv_nrows = [500_000, 2_000_000]
        alphas = [0.05, 0.15, 0.30]
    else:
        nrows_list = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
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
    print(f"Parallel jobs: {args.n_jobs} ({'all CPUs' if args.n_jobs == -1 else ''})")
    print()

    total_t0 = time.time()

    primary_idx = dataset_indices[0]

    if 1 in args.experiments:
        experiment_scale_visibility_parallel(
            dataset_index=primary_idx,
            nrows_list=nrows_list,
            visibilities=visibilities,
            methods=ALL_METHODS,
            n_incidents=n_incidents,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )

    if 2 in args.experiments:
        all_comparison_dfs = []
        for ds_idx in dataset_indices:
            result = experiment_full_comparison_parallel(
                dataset_index=ds_idx,
                visibilities=visibilities,
                methods=ALL_METHODS,
                n_incidents=n_incidents,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
            checkpoint = RESULTS_DIR / f"full_comparison_metrics_CG{ds_idx}.csv"
            result.to_csv(checkpoint, index=False)
            all_comparison_dfs.append(result)

        if len(all_comparison_dfs) > 1:
            merged = pd.concat(all_comparison_dfs, ignore_index=True)
            out = RESULTS_DIR / "multi_topology_metrics.csv"
            merged.to_csv(out, index=False)
            print(f"\nSaved multi-topology results: {out}")

    if 3 in args.experiments:
        experiment_computational_parallel(
            dataset_index=primary_idx,
            nrows_list=comp_nrows,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )

    if 4 in args.experiments:
        experiment_convergence_parallel(
            dataset_index=primary_idx,
            nrows_list=conv_nrows,
            alphas=alphas,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )

    if 5 in args.experiments:
        all_incident_dfs = []
        for ds_idx in dataset_indices:
            result = experiment_per_incident_parallel(
                dataset_index=ds_idx,
                visibilities=visibilities,
                methods=ALL_METHODS,
                n_incidents=n_incidents,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
            checkpoint = RESULTS_DIR / f"per_incident_ranks_CG{ds_idx}.csv"
            result.to_csv(checkpoint, index=False)
            all_incident_dfs.append(result)

        if len(all_incident_dfs) > 1:
            merged = pd.concat(all_incident_dfs, ignore_index=True)
            out = RESULTS_DIR / "multi_topology_per_incident.csv"
            merged.to_csv(out, index=False)
            print(f"\nSaved multi-topology per-incident results: {out}")

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"All experiments completed in {total_elapsed:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
