"""Statistical significance tests for RCA method comparison.

Provides Wilcoxon signed-rank tests, Friedman omnibus test,
Cliff's delta effect sizes, and Nemenyi post-hoc analysis
for paired per-incident comparisons.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


# ── Pairwise Wilcoxon signed-rank ──────────────────────────────────────


def pairwise_wilcoxon(
    per_incident_df: pd.DataFrame,
    proposed: str,
    baselines: list[str],
    metric: str = "rank",
    alternative: str = "less",
) -> pd.DataFrame:
    """Wilcoxon signed-rank test: proposed vs each baseline.

    Parameters
    ----------
    per_incident_df : pd.DataFrame
        Must contain columns [incident_id, visibility, method, <metric>].
    proposed : str
        Name of the proposed method (e.g., "eIRWR").
    baselines : list[str]
        Baseline method names to compare against.
    metric : str
        Column to compare (default "rank"; lower is better).
    alternative : str
        "less" tests if proposed ranks are significantly lower (better).

    Returns
    -------
    pd.DataFrame with columns [visibility, baseline, statistic, p_value,
                                effect_size_r, significant_001, significant_005].
    """
    rows = []
    visibilities = sorted(per_incident_df["visibility"].unique())

    for vis in visibilities:
        vis_df = per_incident_df[per_incident_df["visibility"] == vis]

        proposed_vals = (
            vis_df[vis_df["method"] == proposed]
            .sort_values("incident_id")[metric]
            .values
        )

        for baseline in baselines:
            baseline_vals = (
                vis_df[vis_df["method"] == baseline]
                .sort_values("incident_id")[metric]
                .values
            )

            n = min(len(proposed_vals), len(baseline_vals))
            if n == 0:
                continue

            x = proposed_vals[:n]
            y = baseline_vals[:n]

            # Skip if all differences are zero
            diffs = x - y
            if np.all(diffs == 0):
                rows.append({
                    "visibility": vis,
                    "baseline": baseline,
                    "statistic": np.nan,
                    "p_value": 1.0,
                    "effect_size_r": 0.0,
                    "significant_001": False,
                    "significant_005": False,
                })
                continue

            stat, p = wilcoxon(x, y, alternative=alternative)
            # Effect size r = Z / sqrt(N)
            z = abs((stat - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24))
            r = z / np.sqrt(n)

            rows.append({
                "visibility": vis,
                "baseline": baseline,
                "statistic": stat,
                "p_value": p,
                "effect_size_r": r,
                "significant_001": p < 0.001,
                "significant_005": p < 0.05,
            })

    return pd.DataFrame(rows)


# ── Friedman omnibus test ──────────────────────────────────────────────


def friedman_test(
    per_incident_df: pd.DataFrame,
    methods: list[str],
    metric: str = "rank",
) -> pd.DataFrame:
    """Friedman test across all methods per visibility level.

    Returns
    -------
    pd.DataFrame with [visibility, chi2, p_value, significant].
    """
    rows = []
    visibilities = sorted(per_incident_df["visibility"].unique())

    for vis in visibilities:
        vis_df = per_incident_df[per_incident_df["visibility"] == vis]

        groups = []
        for method in methods:
            vals = (
                vis_df[vis_df["method"] == method]
                .sort_values("incident_id")[metric]
                .values
            )
            groups.append(vals)

        # Align lengths
        min_len = min(len(g) for g in groups)
        if min_len < 3:
            continue
        groups = [g[:min_len] for g in groups]

        chi2, p = friedmanchisquare(*groups)
        rows.append({
            "visibility": vis,
            "chi2": chi2,
            "p_value": p,
            "significant": p < 0.05,
        })

    return pd.DataFrame(rows)


# ── Cliff's delta effect size ──────────────────────────────────────────


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: non-parametric effect size in [-1, 1].

    Positive delta means x tends to be larger than y.
    For ranks (lower = better), negative delta means x is better.
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    more = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1
    return (more - less) / (n_x * n_y)


def compute_effect_sizes(
    per_incident_df: pd.DataFrame,
    proposed: str,
    baselines: list[str],
    metric: str = "rank",
) -> pd.DataFrame:
    """Cliff's delta effect size for each (proposed, baseline) pair.

    Returns
    -------
    pd.DataFrame with [visibility, baseline, cliffs_delta, magnitude].
    """
    rows = []
    visibilities = sorted(per_incident_df["visibility"].unique())

    for vis in visibilities:
        vis_df = per_incident_df[per_incident_df["visibility"] == vis]

        proposed_vals = (
            vis_df[vis_df["method"] == proposed]
            .sort_values("incident_id")[metric]
            .values
        )

        for baseline in baselines:
            baseline_vals = (
                vis_df[vis_df["method"] == baseline]
                .sort_values("incident_id")[metric]
                .values
            )

            n = min(len(proposed_vals), len(baseline_vals))
            if n == 0:
                continue

            d = cliffs_delta(proposed_vals[:n], baseline_vals[:n])
            # Magnitude thresholds (Romano et al., 2006)
            abs_d = abs(d)
            if abs_d < 0.147:
                mag = "negligible"
            elif abs_d < 0.33:
                mag = "small"
            elif abs_d < 0.474:
                mag = "medium"
            else:
                mag = "large"

            rows.append({
                "visibility": vis,
                "baseline": baseline,
                "cliffs_delta": d,
                "magnitude": mag,
            })

    return pd.DataFrame(rows)


# ── Nemenyi post-hoc test ──────────────────────────────────────────────


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference.

    CD = q_alpha * sqrt(k * (k + 1) / (6 * n))

    Uses q_alpha values from Demsar (2006) for common k values.
    """
    # q_alpha for alpha=0.05, from Demsar (2006) Table 5
    q_table_005 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
        14: 3.354,
    }
    q_table_01 = {
        2: 2.576, 3: 2.913, 4: 3.113, 5: 3.255,
        6: 3.364, 7: 3.452, 8: 3.526, 9: 3.590,
        10: 3.646, 11: 3.696, 12: 3.741, 13: 3.781,
        14: 3.818,
    }
    table = q_table_01 if alpha <= 0.01 else q_table_005
    q = table.get(k, 3.354)  # fallback to k=14 value
    return q * np.sqrt(k * (k + 1) / (6 * n))


def nemenyi_posthoc(
    per_incident_df: pd.DataFrame,
    methods: list[str],
    metric: str = "rank",
) -> pd.DataFrame:
    """Nemenyi post-hoc test with average ranks for CD diagram.

    Returns
    -------
    pd.DataFrame with [visibility, method, avg_rank, cd_05].
    """
    rows = []
    visibilities = sorted(per_incident_df["visibility"].unique())

    for vis in visibilities:
        vis_df = per_incident_df[per_incident_df["visibility"] == vis]

        # Build matrix: incidents x methods
        pivot = vis_df.pivot_table(
            index="incident_id", columns="method", values=metric, aggfunc="first",
        )
        available = [m for m in methods if m in pivot.columns]
        pivot = pivot[available].dropna()

        if len(pivot) < 3 or len(available) < 2:
            continue

        # Rank methods per incident (lower metric = better = rank 1)
        ranks = pivot.rank(axis=1, method="average")
        avg_ranks = ranks.mean()

        k = len(available)
        n = len(pivot)
        cd = nemenyi_cd(k, n)

        for method in available:
            rows.append({
                "visibility": vis,
                "method": method,
                "avg_rank": avg_ranks[method],
                "cd_05": cd,
                "n_incidents": n,
            })

    return pd.DataFrame(rows)


# ── Convenience: run all tests ─────────────────────────────────────────


def run_all_tests(
    per_incident_df: pd.DataFrame,
    proposed_methods: list[str] | None = None,
    baseline_methods: list[str] | None = None,
    metric: str = "rank",
) -> dict[str, pd.DataFrame]:
    """Run full statistical analysis suite.

    Parameters
    ----------
    per_incident_df : pd.DataFrame
        Per-incident results with columns [incident_id, visibility, method, rank, ...].
    proposed_methods : list[str], optional
        Methods to test as "proposed". Default: ["eIRWR"].
    baseline_methods : list[str], optional
        Methods to compare against. Default: all other methods.

    Returns
    -------
    dict with keys: "friedman", "wilcoxon", "effect_sizes", "nemenyi".
    """
    all_methods = per_incident_df["method"].unique().tolist()

    if proposed_methods is None:
        proposed_methods = [m for m in ["eIRWR"] if m in all_methods]
    if baseline_methods is None:
        baseline_methods = [m for m in all_methods if m not in proposed_methods]

    results = {}

    # Friedman omnibus
    results["friedman"] = friedman_test(per_incident_df, all_methods, metric)

    # Pairwise Wilcoxon for each proposed method
    wilcoxon_dfs = []
    effect_dfs = []
    for proposed in proposed_methods:
        w = pairwise_wilcoxon(per_incident_df, proposed, baseline_methods, metric)
        w["proposed"] = proposed
        wilcoxon_dfs.append(w)

        e = compute_effect_sizes(per_incident_df, proposed, baseline_methods, metric)
        e["proposed"] = proposed
        effect_dfs.append(e)

    results["wilcoxon"] = pd.concat(wilcoxon_dfs, ignore_index=True) if wilcoxon_dfs else pd.DataFrame()
    results["effect_sizes"] = pd.concat(effect_dfs, ignore_index=True) if effect_dfs else pd.DataFrame()

    # Nemenyi post-hoc
    results["nemenyi"] = nemenyi_posthoc(per_incident_df, all_methods, metric)

    return results
