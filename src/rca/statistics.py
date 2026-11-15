"""Statistical significance tests for RCA method comparison.

Provides the Friedman omnibus test and pairwise Wilcoxon signed-rank
tests for paired per-incident comparisons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

# ── pairwise Wilcoxon signed-rank ──────────────────────────────────────


def pairwise_wilcoxon(
    per_incident_df: pd.DataFrame,
    proposed: str,
    baselines: list[str],
    metric: str = "rank",
    alternative: str = "two-sided",
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
        "two-sided" (default) tests for a difference in either direction;
        "less" tests if proposed ranks are significantly lower (better).

    Returns
    -------
    pd.DataFrame with columns [visibility, baseline, statistic, p_value,
    effect_size_r, proposed_better, significant_001, significant_005,
    n_wins, n_ties, n_losses].
    ``proposed_better`` is True when the proposed method has the lower
    (better) values, i.e. the negative differences carry the larger rank
    sum. It gives the direction of the effect that ``p_value`` tests.
    ``n_wins``, ``n_ties`` and ``n_losses`` count the incidents where the
    proposed value is lower, equal, or higher. ``effect_size_r`` is at
    most about 0.87, reached when there are no losses.
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

            diffs = x - y
            counts = {
                "n_wins": int(np.sum(diffs < 0)),
                "n_ties": int(np.sum(diffs == 0)),
                "n_losses": int(np.sum(diffs > 0)),
            }
            # skip if all differences are zero
            if np.all(diffs == 0):
                rows.append(
                    {
                        "visibility": vis,
                        "baseline": baseline,
                        "statistic": np.nan,
                        "p_value": 1.0,
                        "effect_size_r": 0.0,
                        "proposed_better": False,
                        "significant_001": False,
                        "significant_005": False,
                        **counts,
                    }
                )
                continue

            stat, p = wilcoxon(x, y, alternative=alternative)
            # effect size r = |Z| / sqrt(N), where Z is the tie-corrected
            # normal approximation and N counts the non-zero differences
            # (scipy drops zero differences before ranking)
            z = wilcoxon(x, y, alternative=alternative, method="approx")
            r = abs(z.zstatistic) / np.sqrt(np.count_nonzero(diffs))
            # direction of the effect: compare the rank sums of the negative
            # differences (proposed lower) and the positive ones
            nonzero = diffs[diffs != 0]
            abs_ranks = rankdata(np.abs(nonzero))
            proposed_better = bool(
                abs_ranks[nonzero < 0].sum() > abs_ranks[nonzero > 0].sum()
            )

            rows.append(
                {
                    "visibility": vis,
                    "baseline": baseline,
                    "statistic": stat,
                    "p_value": p,
                    "effect_size_r": r,
                    "proposed_better": proposed_better,
                    "significant_001": p < 0.001,
                    "significant_005": p < 0.05,
                    **counts,
                }
            )

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

        # align lengths
        min_len = min(len(g) for g in groups)
        if min_len < 3:
            continue
        groups = [g[:min_len] for g in groups]

        chi2, p = friedmanchisquare(*groups)
        rows.append(
            {
                "visibility": vis,
                "chi2": chi2,
                "p_value": p,
                "significant": p < 0.05,
            }
        )

    return pd.DataFrame(rows)


# ── convenience: run all tests ─────────────────────────────────────────


def run_all_tests(
    per_incident_df: pd.DataFrame,
    proposed: str = "eIRWR",
    metric: str = "rank",
) -> dict[str, pd.DataFrame]:
    """Run the Friedman test and the pairwise Wilcoxon tests.

    Parameters
    ----------
    per_incident_df : pd.DataFrame
        Per-incident results with columns
        [incident_id, visibility, method, rank, ...].
    proposed : str
        Method compared against every other method.
    metric : str
        Column to compare (default "rank"; lower is better).

    Returns
    -------
    dict with keys "friedman" and "wilcoxon".
    """
    all_methods = per_incident_df["method"].unique().tolist()
    baselines = [m for m in all_methods if m != proposed]

    wilcoxon_df = pairwise_wilcoxon(
        per_incident_df, proposed, baselines, metric
    )
    wilcoxon_df["proposed"] = proposed

    return {
        "friedman": friedman_test(per_incident_df, all_methods, metric),
        "wilcoxon": wilcoxon_df,
    }
