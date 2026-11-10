"""Run the experiments reported in the paper.

Experiments (select with ``--experiments``):

  comparison   Simulated incidents on each topology, all methods.
               Writes per_incident_ranks_CG{i}.csv and
               full_comparison_metrics_CG{i}.csv (Table main-results,
               Figs. mrr-visibility and per-topology), plus
               multi_topology_metrics.csv when several topologies run.
  compute      Runtime and peak memory vs graph size on prefixes of the
               primary topology (Table computational, Fig. scalability).
               Methods are timed one at a time, never in parallel.
  convergence  IRWR/eIRWR iterations vs restart probability
               (Fig. convergence).
  stats        Friedman, Wilcoxon and Cliff's delta on the per-incident
               ranks of the primary topology (Table wilcoxon). The
               trace-level methods are left out, as in the paper.
  real         Label-free evaluation on real anomaly windows
               (Table real-world). Run scripts/analyze_real_metrics.py
               first.

Every method sees the same incidents. The root causes come from
``default_rng(seed)`` and the noise of incident ``i`` from
``default_rng(seed * 100000 + i)``, so results do not depend on
``--n-jobs``.

Usage:
    uv run python scripts/evaluate_performance.py
    uv run python scripts/evaluate_performance.py --experiments comparison
    uv run python scripts/evaluate_performance.py --quick --n-jobs 8
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import DATA_ROOT, load_call_graph_by_index  # noqa: E402
from src.rca.baselines import (  # noqa: E402
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
from src.rca.graph import (  # noqa: E402
    build_dependency_graph,
    compute_fault_propagation_matrix,
    compute_weight_matrix,
)
from src.rca.irwr import irwr, irwr_enhanced  # noqa: E402
from src.rca.metrics import (  # noqa: E402
    average_precision,
    get_rank,
    mean_reciprocal_rank,
    precision_at_k,
    precision_at_k_value,
    recall_at_k,
)
from src.rca.simulation import simulate_anomaly  # noqa: E402
from src.rca.tracerank import simulate_trace_data, tracerank  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"

# "CORAL" is shown as "Corr-Fusion" in the paper
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
# trace-level methods consume traces simulated from the true root cause,
# so they cannot run on real anomaly windows
TRACE_METHODS = {"MicroRank", "TraceRank"}

R_BASE = 0.1
ALPHA = 0.15


# ── helpers ─────────────────────────────────────────────────────────────


def prepare_graph(dataset_index: int, nrows: int | None = None) -> dict:
    """Load a CallGraph file and build W and M_R = diag(1-R) W."""
    df = load_call_graph_by_index(dataset_index, nrows=nrows)
    nodes, edges_df = build_dependency_graph(df)
    W = compute_weight_matrix(edges_df, nodes)
    M_R = compute_fault_propagation_matrix(W, R=R_BASE).T.tocsr()
    node_to_idx = {name: i for i, name in enumerate(nodes)}
    callee_indices = sorted(node_to_idx[dm] for dm in edges_df["dm"].unique())
    return {
        "nodes": nodes,
        "W": W,
        "M_R": M_R,
        "callee_indices": callee_indices,
        "n_rows": len(df),
        "n_nodes": len(nodes),
        "n_edges": len(edges_df),
    }


def graph_meta(graph: dict, dataset_index: int) -> dict:
    return {
        "dataset_index": dataset_index,
        "n_rows": graph["n_rows"],
        "n_nodes": graph["n_nodes"],
        "n_edges": graph["n_edges"],
    }


def compute_all_metrics(ranked: np.ndarray, root_idx: int) -> dict:
    """Metric suite for one incident."""
    out: dict = {"rank": get_rank(ranked, root_idx)}
    for k in (1, 3, 5):
        out[f"PR@{k}"] = precision_at_k(ranked, root_idx, k)
    for k in (1, 3, 5):
        out[f"Precision@{k}"] = precision_at_k_value(ranked, root_idx, k)
    for k in (1, 3, 5):
        out[f"Recall@{k}"] = recall_at_k(ranked, root_idx, k)
    out["AP"] = average_precision(ranked, root_idx)
    return out


def aggregate_metrics(incidents: pd.DataFrame) -> dict:
    """Average per-incident metrics of one (visibility, method) group."""
    cols = [
        f"{m}@{k}" for m in ("PR", "Precision", "Recall") for k in (1, 3, 5)
    ]
    out = {c: incidents[c].mean() for c in cols}
    out["MRR"] = mean_reciprocal_rank(incidents["rank"].tolist())
    out["MAP"] = incidents["AP"].mean()
    out["avg_rank"] = incidents["rank"].mean()
    out["median_rank"] = incidents["rank"].median()
    return out


def run_method(
    method: str,
    W,
    M_R,
    s_obs: np.ndarray,
    root_idx: int,
    visibility: float,
    seed: int,
    incident_idx: int,
) -> np.ndarray:
    """Run one RCA method and return its score vector."""
    if method == "MicroRCA":
        return personalized_pagerank(W, s_obs, alpha=ALPHA)
    if method == "CloudRanger":
        n_walks = 50000 if W.shape[0] > 10000 else 10000
        return second_order_rw(
            W,
            s_obs,
            n_walks=n_walks,
            walk_length=20,
            rng=np.random.default_rng(seed * 10000 + incident_idx),
        )
    if method == "In-Degree":
        return degree_centrality(W)
    if method == "MicroHECL":
        return microhecl(W, s_obs)
    if method == "TraceDiag":
        return tracediag(W, s_obs)
    if method == "Neighbor Corr.":
        return neighbor_correlation(W, s_obs)
    if method == "MonitorRank":
        return monitor_rank(W, s_obs)
    if method == "CORAL":
        return coral_rca(W, s_obs)
    if method == "IRWR":
        return irwr(M_R, s_obs, alpha=ALPHA)[0]
    if method == "eIRWR":
        return irwr_enhanced(W, s_obs)[0]
    if method in TRACE_METHODS:
        traces, is_anom, fe_idx = simulate_trace_data(
            W,
            root_idx,
            root_cause_visibility=visibility,
            rng=np.random.default_rng(seed * 10000 + incident_idx),
        )
        if method == "TraceRank":
            return tracerank(W, traces, is_anom, frontend_idx=fe_idx)
        return micro_rank(W, s_obs, traces, is_anom)
    raise ValueError(f"Unknown method: {method}")


# ── comparison on simulated incidents ──────────────────────────────────


def _run_incident(
    methods: list[str],
    graph: dict,
    root_idx: int,
    vis: float,
    seed: int,
    incident_idx: int,
) -> list[dict]:
    """Simulate one incident and rank it with every method."""
    rng = np.random.default_rng(seed * 100000 + incident_idx)
    s_obs = simulate_anomaly(
        graph["W"],
        root_idx,
        propagation_steps=5,
        decay=0.6,
        noise_level=0.05,
        root_cause_visibility=vis,
        rng=rng,
    )
    records = []
    for method in methods:
        scores = run_method(
            method,
            graph["W"],
            graph["M_R"],
            s_obs,
            root_idx,
            vis,
            seed,
            incident_idx,
        )
        records.append(
            {
                "incident_id": incident_idx,
                "visibility": vis,
                "method": method,
                "root_cause_idx": root_idx,
                "root_cause_node": graph["nodes"][root_idx],
                **compute_all_metrics(np.argsort(-scores), root_idx),
            }
        )
    return records


def experiment_comparison(
    dataset_index: int,
    visibilities: list[float],
    methods: list[str],
    n_incidents: int,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """All methods on simulated incidents over the full topology."""
    print(f"\n{'=' * 70}\nCOMPARISON: CallGraph_{dataset_index}\n{'=' * 70}")
    t0 = time.time()
    graph = prepare_graph(dataset_index)
    print(
        f"  {graph['n_nodes']:,} nodes, {graph['n_edges']:,} edges"
        f" ({time.time() - t0:.1f}s)"
    )

    per_incident = []
    for vis in visibilities:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        roots = [
            rng.choice(graph["callee_indices"]) for _ in range(n_incidents)
        ]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_incident)(methods, graph, root, vis, seed, i)
            for i, root in enumerate(roots)
        )
        per_incident.extend(rec for recs in results for rec in recs)
        print(f"  vis={vis:.2f}: {time.time() - t0:.1f}s")

    per_df = pd.DataFrame(per_incident)
    per_df.to_csv(
        RESULTS_DIR / f"per_incident_ranks_CG{dataset_index}.csv", index=False
    )

    summary = []
    for (vis, method), grp in per_df.groupby(
        ["visibility", "method"], sort=False
    ):
        summary.append(
            {
                "visibility": vis,
                "method": method,
                "n_incidents": len(grp),
                **aggregate_metrics(grp),
                **graph_meta(graph, dataset_index),
            }
        )
    summary_df = pd.DataFrame(summary)
    out = RESULTS_DIR / f"full_comparison_metrics_CG{dataset_index}.csv"
    summary_df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    return summary_df


# ── computational cost ─────────────────────────────────────────────────


def bench_time(fn, n_repeats: int) -> tuple[float, float]:
    """Return (mean, std) wall-clock seconds over ``n_repeats`` calls."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def bench_memory(fn) -> int:
    """Return peak traced memory (bytes) during one call of ``fn``."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _bench_fns(W, M_R, s_obs, traces, is_anom, fe_idx) -> dict:
    """Zero-argument callables, one per method, for benchmarking."""
    n_walks = 50000 if W.shape[0] > 10000 else 10000
    return {
        "MicroRCA": lambda: personalized_pagerank(W, s_obs, alpha=ALPHA),
        "CloudRanger": lambda: second_order_rw(
            W,
            s_obs,
            n_walks=n_walks,
            walk_length=20,
            rng=np.random.default_rng(0),
        ),
        "In-Degree": lambda: degree_centrality(W),
        "MicroHECL": lambda: microhecl(W, s_obs),
        "TraceDiag": lambda: tracediag(W, s_obs),
        "Neighbor Corr.": lambda: neighbor_correlation(W, s_obs),
        "MonitorRank": lambda: monitor_rank(W, s_obs),
        "MicroRank": lambda: micro_rank(W, s_obs, traces, is_anom),
        "CORAL": lambda: coral_rca(W, s_obs),
        "TraceRank": lambda: tracerank(W, traces, is_anom, frontend_idx=fe_idx),
        "IRWR": lambda: irwr(M_R, s_obs, alpha=ALPHA),
        "eIRWR": lambda: irwr_enhanced(W, s_obs),
    }


def experiment_compute(
    dataset_index: int,
    nrows_list: list[int],
    methods: list[str],
    seed: int,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Runtime and peak memory per method on growing graph prefixes."""
    print(f"\n{'=' * 70}\nCOMPUTE: CallGraph_{dataset_index}\n{'=' * 70}")
    records = []
    for nrows in nrows_list:
        graph = prepare_graph(dataset_index, nrows=nrows)
        W, M_R = graph["W"], graph["M_R"]
        print(f"\n  nrows={nrows:,}: {graph['n_nodes']:,} nodes")

        rng = np.random.default_rng(seed)
        root_idx = rng.choice(graph["callee_indices"])
        s_obs = simulate_anomaly(
            W, root_idx, root_cause_visibility=0.3, rng=rng
        )
        traces, is_anom, fe_idx = simulate_trace_data(
            W, root_idx, rng=np.random.default_rng(seed)
        )
        bench = _bench_fns(W, M_R, s_obs, traces, is_anom, fe_idx)
        for method in methods:
            fn = bench[method]
            reps = 3 if method == "CloudRanger" else n_repeats
            mean_t, std_t = bench_time(fn, reps)
            peak = bench_memory(fn)
            records.append(
                {
                    "dataset_index": dataset_index,
                    "nrows": nrows,
                    "n_nodes": graph["n_nodes"],
                    "n_edges": graph["n_edges"],
                    "method": method,
                    "mean_time_s": mean_t,
                    "std_time_s": std_t,
                    "peak_memory_bytes": peak,
                    "peak_memory_mb": peak / (1024 * 1024),
                }
            )
            print(
                f"    {method:15s} time={mean_t * 1000:10.2f} ms"
                f"  mem={peak / 1024**2:8.2f} MB"
            )

    df = pd.DataFrame(records)
    out = RESULTS_DIR / "computational_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    return df


# ── convergence ────────────────────────────────────────────────────────


def experiment_convergence(
    dataset_index: int,
    nrows_list: list[int],
    alphas: list[float],
    seed: int,
    n_trials: int = 20,
) -> pd.DataFrame:
    """Iterations to convergence of IRWR and eIRWR vs restart alpha."""
    print(f"\n{'=' * 70}\nCONVERGENCE: CallGraph_{dataset_index}\n{'=' * 70}")
    records = []
    for nrows in nrows_list:
        graph = prepare_graph(dataset_index, nrows=nrows)
        W, M_R = graph["W"], graph["M_R"]
        print(f"\n  nrows={nrows:,}: {graph['n_nodes']:,} nodes")
        for alpha in alphas:
            rng = np.random.default_rng(seed)
            iters: dict[str, list[int]] = {"IRWR": [], "eIRWR": []}
            for _ in range(n_trials):
                root_idx = rng.choice(graph["callee_indices"])
                s_obs = simulate_anomaly(
                    W, root_idx, root_cause_visibility=0.3, rng=rng
                )
                iters["IRWR"].append(irwr(M_R, s_obs, alpha=alpha)[1])
                iters["eIRWR"].append(irwr_enhanced(W, s_obs, alpha=alpha)[1])
            for method, its in iters.items():
                records.append(
                    {
                        "dataset_index": dataset_index,
                        "nrows": nrows,
                        "n_nodes": graph["n_nodes"],
                        "n_edges": graph["n_edges"],
                        "alpha": alpha,
                        "method": method,
                        "mean_iters": np.mean(its),
                        "std_iters": np.std(its),
                        "max_iters": np.max(its),
                        "min_iters": np.min(its),
                    }
                )
            print(
                f"    alpha={alpha:.2f}: IRWR={np.mean(iters['IRWR']):.1f}, "
                f"eIRWR={np.mean(iters['eIRWR']):.1f} iterations"
            )

    df = pd.DataFrame(records)
    out = RESULTS_DIR / "convergence_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    return df


# ── statistical tests ──────────────────────────────────────────────────


def experiment_stats(dataset_index: int) -> None:
    """Friedman, Wilcoxon and Cliff's delta on per-incident ranks."""
    from src.rca.statistics import run_all_tests

    print(f"\n{'=' * 70}\nSTATS: CallGraph_{dataset_index}\n{'=' * 70}")
    path = RESULTS_DIR / f"per_incident_ranks_CG{dataset_index}.csv"
    if not path.exists():
        print(f"  ERROR: {path} not found. Run the comparison first.")
        return
    df = pd.read_csv(path)
    # the trace-level methods are not part of the paper's comparison, so the
    # tests cover the aggregate-metric methods only (as in experiment_real)
    df = df[~df["method"].isin(TRACE_METHODS)]
    for name, res in run_all_tests(df).items():
        out = RESULTS_DIR / f"statistical_{name}.csv"
        res.to_csv(out, index=False)
        print(f"  Saved: {out}")
        if name == "wilcoxon":
            cols = ["baseline", "p_value", "effect_size_r", "proposed_better"]
            print(res[res["visibility"] == 0.3][cols].to_string(index=False))


# ── real anomaly windows ───────────────────────────────────────────────


def experiment_real(
    dataset_index: int,
    methods: list[str],
    seed: int,
    n_windows: int = 50,
) -> pd.DataFrame:
    """Label-free plausibility of each method on real anomaly windows."""
    from src.loader import load_msrt_mcr
    from src.rca.real_eval import (
        build_real_anomaly_vector,
        structural_plausibility_score,
    )

    print(f"\n{'=' * 70}\nREAL: CallGraph_{dataset_index}\n{'=' * 70}")
    anomaly_path = RESULTS_DIR / "real_anomaly_windows.csv"
    msrtmcr_path = DATA_ROOT / "MSRTMCR" / "MCRRTUpdate_0.csv"
    for p in (anomaly_path, msrtmcr_path):
        if not p.exists():
            print(
                f"  ERROR: {p} not found (see scripts/analyze_real_metrics.py)"
            )
            return pd.DataFrame()

    graph = prepare_graph(dataset_index)
    timestamps = sorted(pd.read_csv(anomaly_path)["timestamp"].unique())
    if len(timestamps) > n_windows:
        rng = np.random.default_rng(seed)
        timestamps = sorted(rng.choice(timestamps, n_windows, replace=False))
    msrtmcr_df = load_msrt_mcr(path=msrtmcr_path)
    print(f"  {len(timestamps)} windows, {len(msrtmcr_df):,} MSRTMCR rows")

    methods = [m for m in methods if m not in TRACE_METHODS]
    records = []
    for i, ts in enumerate(timestamps):
        s_obs = build_real_anomaly_vector(msrtmcr_df, graph["nodes"], ts)
        if s_obs.max() < 1e-6:
            continue
        for method in methods:
            scores = run_method(
                method,
                graph["W"],
                graph["M_R"],
                s_obs,
                root_idx=-1,
                visibility=0.0,
                seed=seed,
                incident_idx=i,
            )
            ranked = np.argsort(-scores)
            records.append(
                {
                    "timestamp": ts,
                    "method": method,
                    "top_1": graph["nodes"][ranked[0]],
                    **structural_plausibility_score(
                        ranked, graph["W"], s_obs, k=5
                    ),
                }
            )

    df = pd.DataFrame(records)
    out = RESULTS_DIR / "real_world_evaluation.csv"
    df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    if len(df):
        print(
            df.pivot(
                index="method", columns="timestamp", values="cascade_coverage"
            )
            .round(3)
            .to_string()
        )
    return df


# ── main ───────────────────────────────────────────────────────────────

EXPERIMENTS = ["comparison", "compute", "convergence", "stats", "real"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset-indices",
        nargs="+",
        type=int,
        default=[84, 136, 297],
        help="CallGraph indices for the comparison (default: 84 136 297)",
    )
    parser.add_argument(
        "--primary-index",
        type=int,
        default=297,
        help="Topology for compute, convergence, stats and real (default: 297)",
    )
    parser.add_argument("--n-incidents", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=EXPERIMENTS,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for the comparison (default: all CPUs)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer visibilities, incidents, sizes and alphas",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip comparison topologies whose result CSV already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.quick:
        visibilities = [0.0, 0.1, 0.3, 0.5]
        n_incidents = min(args.n_incidents, 50)
        comp_nrows = [500_000, 2_000_000]
        conv_nrows = [500_000, 2_000_000]
        alphas = [0.05, 0.15, 0.30]
    else:
        visibilities = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        n_incidents = args.n_incidents
        comp_nrows = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
        conv_nrows = [500_000, 1_000_000, 2_000_000, 5_000_000]
        alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    print(f"Experiments: {args.experiments}")
    print(f"Topologies: {args.dataset_indices}, primary {args.primary_index}")
    print(f"Incidents: {n_incidents}, visibilities: {visibilities}")
    t_start = time.time()

    if "comparison" in args.experiments:
        frames = []
        for idx in args.dataset_indices:
            ckpt = RESULTS_DIR / f"full_comparison_metrics_CG{idx}.csv"
            if args.resume and ckpt.exists():
                print(f"\n  Resuming: {ckpt.name} exists, skipping")
                frames.append(pd.read_csv(ckpt))
                continue
            frames.append(
                experiment_comparison(
                    idx,
                    visibilities,
                    ALL_METHODS,
                    n_incidents,
                    args.seed,
                    args.n_jobs,
                )
            )
        if len(frames) > 1:
            out = RESULTS_DIR / "multi_topology_metrics.csv"
            pd.concat(frames, ignore_index=True).to_csv(out, index=False)
            print(f"\nSaved: {out}")

    if "compute" in args.experiments:
        experiment_compute(
            args.primary_index, comp_nrows, ALL_METHODS, args.seed
        )

    if "convergence" in args.experiments:
        experiment_convergence(
            args.primary_index, conv_nrows, alphas, args.seed
        )

    if "stats" in args.experiments:
        experiment_stats(args.primary_index)

    if "real" in args.experiments:
        experiment_real(args.primary_index, ALL_METHODS, args.seed)

    print(f"\nAll experiments completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
