#!/usr/bin/env python3
"""Statistical tests for the results tables."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import scipy functions explicitly so older scipy versions with lazy-loading
# don't hide them behind the stats namespace.
from scipy import stats
from scipy.stats import rankdata, shapiro, ttest_rel, wilcoxon

# scipy.stats.binomtest was added in 1.7.0.  For older installs we provide a
# pure-Python fallback that requires only math.comb (Python >= 3.8).
# We only need the two-sided, p = 0.5 case (exact McNemar / sign test).
import math as _math

def _binom_cdf_exact(k: int, n: int) -> float:
    """P(X <= k) for X ~ Bin(n, 0.5), computed without scipy."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = sum(_math.comb(n, i) for i in range(int(k) + 1))
    return total / (2 ** n)


class _BinomResult:
    def __init__(self, pvalue: float) -> None:
        self.pvalue = pvalue


if not hasattr(stats, "binomtest"):
    def _binomtest(k: int, n: int, p: float = 0.5, alternative: str = "two-sided"):
        if n == 0:
            return _BinomResult(1.0)
        k, n = int(k), int(n)
        if alternative == "two-sided" and p == 0.5:
            pv = float(min(1.0, 2.0 * _binom_cdf_exact(min(k, n - k), n)))
        elif alternative == "greater":
            pv = float(1.0 - _binom_cdf_exact(k - 1, n))
        else:
            pv = float(_binom_cdf_exact(k, n))
        return _BinomResult(pv)

    stats.binomtest = _binomtest  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle_common import (  # noqa: E402
    ALPHA,
    CONSOLIDATED,
    CS_METHODS,
    METRIC_FAMILIES,
    METRIC_HIGHER_BETTER,
    SRG_PHENOTYPE_METHODS,
    STATS,
    Z_975,
    holm_correct,
    order_datasets,
    order_methods,
    record_stage,
    setup_logging,
    wilson_interval,
    write_csv,
    write_text,
)

LOG = setup_logging()

BOOTSTRAP_RESAMPLES = 2000
SEED = 20260729

TESTED_METRICS: list[str] = [
    *METRIC_FAMILIES["proximity_perceptual"],
    *METRIC_FAMILIES["class_reference"],
    *METRIC_FAMILIES["confidence"],
    *METRIC_FAMILIES["embedding"],
    "cmmd_vs_source_y_true_class",
    "cmmd_vs_predicted_class",
]

METRIC_TO_FAMILY: dict[str, str] = {
    metric: family for family, metrics in METRIC_FAMILIES.items() for metric in metrics
}
METRIC_TO_FAMILY.setdefault("cmmd_vs_source_y_true_class", "per_sample_cmmd")
METRIC_TO_FAMILY.setdefault("cmmd_vs_predicted_class", "per_sample_cmmd")

# subset policies for continuous comparisons
SUBSET_POLICIES: dict[str, str] = {
    "intersection_flips": (
        "images where both compared methods flipped the prediction; matches the "
        "protocol behind the p-values currently in the paper"
    ),
    "common_success": (
        "images where every method flipped the prediction; removes the "
        "conditioning advantage that low-validity methods enjoy"
    ),
}

PROTOCOL_TEXT = """\
# Statistical protocol

Validity is a proportion over the shared evaluation manifest. Each method is
summarised by its flip count with a Wilson 95% interval, and methods are compared
with an exact McNemar test on the discordant image pairs. A Wilcoxon signed-rank
test on 0/1 indicators, as used previously, is not a valid test for this design.

Continuous metrics are properties of a counterfactual and therefore only defined
where a method succeeded. Two subset policies are reported. The first pairs two
methods on the images both of them flipped, which is the protocol behind the
p-values currently printed in the paper and gives a different sample for every
pair. The second restricts every comparison to the images that all methods
flipped, which removes the advantage a low-validity method gains by being scored
only on the cases it happened to solve. Ranking means shown in the tables are
computed on each method's own successes, so they differ from the means on the
tested subset; both are reported side by side.

Each comparison uses Shapiro-Wilk on the paired differences at alpha = 0.05 to
choose between a paired t-test and a Wilcoxon signed-rank test, and reports the
paired sample size, the median paired difference with a percentile bootstrap
interval, and an effect size: Cohen's dz for the t-test, matched-pairs rank
biserial correlation for Wilcoxon.

Multiplicity is controlled with the Holm step-down procedure. Two scopes are
given: within each dataset and metric across all method pairs, and within the
comparisons the paper actually reports, one per table cell. The second is the
scope to quote in the paper.

Set-level distributional scores are single numbers per method, so they are ranked
rather than tested, and each is annotated with the size of the counterfactual set
and how many of its members failed to flip.

Deep-SCM results are aggregate means and standard deviations supplied by the
benchmark, so they are ranked without tests.
"""




def _clean_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def bootstrap_median_ci(
    diff: np.ndarray, rng: np.random.Generator, resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float]:
    if diff.size < 3:
        return float("nan"), float("nan")
    idx = rng.integers(0, diff.size, size=(resamples, diff.size))
    medians = np.median(diff[idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank biserial correlation for the Wilcoxon test."""
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return float("nan")
    ranks = rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    total = positive + negative
    return float((positive - negative) / total) if total else float("nan")


def paired_test(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """Shapiro-gated paired test with effect size and bootstrap interval."""
    a, b = _clean_pair(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    n = int(a.size)
    out = {
        "n": n,
        "mean_a_tested": float(a.mean()) if n else np.nan,
        "mean_b_tested": float(b.mean()) if n else np.nan,
        "median_diff": np.nan,
        "diff_ci95_low": np.nan,
        "diff_ci95_high": np.nan,
        "test": "insufficient_n",
        "shapiro_p": np.nan,
        "normal_diff": False,
        "p_raw": np.nan,
        "effect_size": np.nan,
        "effect_size_kind": "",
    }
    if n < 3:
        return out

    diff = a - b
    out["median_diff"] = float(np.median(diff))
    out["diff_ci95_low"], out["diff_ci95_high"] = bootstrap_median_ci(diff, rng)

    if np.allclose(diff, 0.0):
        out["test"] = "identical"
        out["p_raw"] = 1.0
        out["effect_size"] = 0.0
        out["effect_size_kind"] = "none"
        return out

    spread = float(np.std(diff, ddof=1))
    shapiro_p = float(shapiro(diff).pvalue) if spread > 0 else 1.0
    out["shapiro_p"] = shapiro_p
    normal = shapiro_p >= ALPHA
    out["normal_diff"] = bool(normal)

    if normal:
        out["test"] = "paired_t"
        out["p_raw"] = float(ttest_rel(a, b).pvalue)
        out["effect_size"] = float(np.mean(diff) / spread) if spread else np.nan
        out["effect_size_kind"] = "cohens_dz"
    else:
        out["test"] = "wilcoxon"
        try:
            out["p_raw"] = float(
                wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue
            )
        except ValueError:
            out["test"] = "wilcoxon_failed"
        out["effect_size"] = rank_biserial(diff)
        out["effect_size_kind"] = "rank_biserial"
    return out


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test for paired binary outcomes."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = int(a.size)
    b_only = int(np.sum((a == 1) & (b == 0)))
    c_only = int(np.sum((a == 0) & (b == 1)))
    discordant = b_only + c_only
    p = (
        float(stats.binomtest(b_only, discordant, 0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    return {
        "n": n,
        "n_a_only": b_only,
        "n_b_only": c_only,
        "n_discordant": discordant,
        "rate_a": float(a.mean()) if n else np.nan,
        "rate_b": float(b.mean()) if n else np.nan,
        "rate_difference": float(a.mean() - b.mean()) if n else np.nan,
        "test": "mcnemar_exact",
        "p_raw": p,
    }


def _add_holm(df: pd.DataFrame, by: list[str], column: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df[column] = np.nan
    for _, index in df.groupby(by, observed=True, dropna=False).groups.items():
        rows = df.loc[index]
        df.loc[index, column] = holm_correct(rows["p_raw"].tolist())
    df[f"{column}_significant"] = df[column] < ALPHA
    df["p_raw_significant"] = df["p_raw"] < ALPHA
    return df




def validity_tests(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in wide.groupby("dataset", observed=True):
        pivot = g.pivot_table(index="sample_idx", columns="method", values="class_flip", observed=True)
        methods = order_methods(pivot.columns)
        for first, second in itertools.combinations(methods, 2):
            pair = pivot[[first, second]].dropna()
            result = mcnemar_exact(pair[first].to_numpy(), pair[second].to_numpy())
            rows.append(
                {
                    "dataset": dataset,
                    "metric": "class_flip",
                    "method_a": first,
                    "method_b": second,
                    **result,
                }
            )
    out = pd.DataFrame(rows)
    return _add_holm(out, ["dataset"], "p_holm_dataset")


def validity_table(validity: pd.DataFrame) -> pd.DataFrame:
    out = validity.copy()
    out["validity_display"] = [
        f"{int(k)}/{int(n)} [{lo * 100:.1f}, {hi * 100:.1f}]"
        for k, n, lo, hi in zip(
            out["n_class_flip"],
            out["n_attempted"],
            out["validity_wilson_low"],
            out["validity_wilson_high"],
        )
    ]
    out["rank_in_dataset"] = out.groupby("dataset")["validity_rate"].rank(
        ascending=False, method="min"
    )
    return out.sort_values(["dataset", "rank_in_dataset", "method"])




def _subset_frame(wide: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "common_success":
        return wide[wide["in_common_success"] == 1]
    return wide[wide["class_flip"] == 1]


def paired_tests(wide: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for policy in SUBSET_POLICIES:
        frame = _subset_frame(wide, policy)
        for dataset, g in frame.groupby("dataset", observed=True):
            for metric in TESTED_METRICS:
                if metric not in g.columns:
                    continue
                pivot = g.pivot_table(index="sample_idx", columns="method", values=metric, observed=True)
                methods = order_methods(pivot.columns)
                for first, second in itertools.combinations(methods, 2):
                    pair = pivot[[first, second]].dropna()
                    if pair.empty:
                        continue
                    result = paired_test(
                        pair[first].to_numpy(), pair[second].to_numpy(), rng
                    )
                    higher_better = METRIC_HIGHER_BETTER.get(metric)
                    better = None
                    if np.isfinite(result["mean_a_tested"]) and np.isfinite(
                        result["mean_b_tested"]
                    ):
                        a_better = (
                            result["mean_a_tested"] > result["mean_b_tested"]
                            if higher_better
                            else result["mean_a_tested"] < result["mean_b_tested"]
                        )
                        better = first if a_better else second
                    rows.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "metric_family": METRIC_TO_FAMILY.get(metric, "other"),
                            "higher_is_better": higher_better,
                            "subset_policy": policy,
                            "subset_definition": SUBSET_POLICIES[policy],
                            "method_a": first,
                            "method_b": second,
                            "better_by_tested_mean": better,
                            **result,
                        }
                    )
    out = pd.DataFrame(rows)
    out = _add_holm(out, ["dataset", "metric", "subset_policy"], "p_holm_metric")
    out = _add_holm(out, ["dataset", "metric_family", "subset_policy"], "p_holm_family")
    return out




def _pick_competitor(ranked: list[str], is_special) -> tuple[str, str, str]:
    """Best vs runner-up, or best vs best outsider when the top two are ours."""
    first, second = ranked[0], ranked[1]
    if is_special(first) and is_special(second):
        outsiders = [m for m in ranked if not is_special(m)]
        if outsiders:
            return first, outsiders[0], "best_vs_best_external"
    return first, second, "best_vs_second"


def _lookup_test(tests: pd.DataFrame, **keys) -> pd.Series | None:
    """Find the test row for a pair, orienting it as method_a minus method_b."""
    if tests.empty:
        return None

    def _match(mapping: dict) -> pd.DataFrame:
        mask = pd.Series(True, index=tests.index)
        for column, value in mapping.items():
            mask &= tests[column] == value
        return tests[mask]

    subset = _match(keys)
    if not subset.empty:
        row = subset.iloc[0].copy()
        row["swapped"] = False
        return row

    reversed_keys = dict(keys)
    reversed_keys["method_a"], reversed_keys["method_b"] = keys["method_b"], keys["method_a"]
    subset = _match(reversed_keys)
    if subset.empty:
        return None

    row = subset.iloc[0].copy()
    row["swapped"] = True
    for left, right in (("mean_a_tested", "mean_b_tested"), ("rate_a", "rate_b")):
        if left in row and right in row:
            row[left], row[right] = row[right], row[left]
    for column in ("median_diff", "effect_size", "rate_difference"):
        if column in row and pd.notna(row[column]):
            row[column] = -row[column]
    if "diff_ci95_low" in row and "diff_ci95_high" in row:
        low, high = row["diff_ci95_low"], row["diff_ci95_high"]
        row["diff_ci95_low"], row["diff_ci95_high"] = (
            -high if pd.notna(high) else high,
            -low if pd.notna(low) else low,
        )
    return row


def _same_display(first: float, second: float) -> tuple[bool, bool]:
    """Whether two means print identically at two decimals / three significant digits."""
    if not (np.isfinite(first) and np.isfinite(second)):
        return False, False
    return f"{first:.2f}" == f"{second:.2f}", f"{first:.3g}" == f"{second:.3g}"


def _comparison_row(
    *,
    table: str,
    dataset: str,
    metric: str,
    metric_family: str,
    subset_policy: str,
    means: pd.Series,
    ranked: list[str],
    best: str,
    competitor: str,
    rule: str,
    role: str,
    test: pd.Series | None,
    binary: bool = False,
) -> dict:
    tie_2dp, tie_3sf = _same_display(float(means.loc[best]), float(means.loc[competitor]))
    row = {
        "table": table,
        "dataset": dataset,
        "metric": metric,
        "metric_family": metric_family,
        "subset_policy": subset_policy,
        "comparison_role": role,
        "best": best,
        "second": ranked[1] if len(ranked) > 1 else "",
        "competitor": competitor,
        "rule": rule,
        "best_mean_displayed": float(means.loc[best]),
        "competitor_mean_displayed": float(means.loc[competitor]),
        "best_mean_tested": np.nan,
        "competitor_mean_tested": np.nan,
        "tie_at_2dp": tie_2dp,
        "tie_at_3sf": tie_3sf,
        "n": np.nan,
        "test": "missing",
        "p_raw": np.nan,
        "effect_size": np.nan,
        "effect_size_kind": "",
        "test_favours": "",
        "test_agrees_with_ranking": False,
        "verdict": "no_test",
        "ranking": ", ".join(f"{m}:{means.loc[m]:.4g}" for m in ranked),
    }
    if test is None:
        return row

    row["n"] = int(test["n"])
    row["p_raw"] = float(test["p_raw"])
    if binary:
        row["test"] = "mcnemar_exact"
        row["best_mean_tested"] = float(test["rate_a"])
        row["competitor_mean_tested"] = float(test["rate_b"])
        row["effect_size"] = float(test["rate_difference"])
        row["effect_size_kind"] = "rate_difference"
        row["test_favours"] = best if test["rate_difference"] > 0 else competitor
        row["test_agrees_with_ranking"] = bool(
            np.isfinite(test["p_raw"])
            and test["p_raw"] < ALPHA
            and test["rate_difference"] > 0
        )
        row["verdict"] = _verdict(test["p_raw"], row["test_favours"], best, competitor)
        return row

    row["test"] = str(test["test"])
    row["best_mean_tested"] = float(test["mean_a_tested"])
    row["competitor_mean_tested"] = float(test["mean_b_tested"])
    row["effect_size"] = float(test["effect_size"])
    row["effect_size_kind"] = str(test["effect_size_kind"])
    row["test_favours"] = str(test["better_by_tested_mean"])
    row["test_agrees_with_ranking"] = bool(
        np.isfinite(test["p_raw"])
        and test["p_raw"] < ALPHA
        and test["better_by_tested_mean"] == best
    )
    row["verdict"] = _verdict(test["p_raw"], row["test_favours"], best, competitor)
    return row


def _verdict(p: float, favours: str, best: str, competitor: str) -> str:
    """Disambiguate "not significant" from "significant the other way round"."""
    if not np.isfinite(p):
        return "no_test"
    if p >= ALPHA:
        return "not_significant"
    if favours == best:
        return f"significant_for_{best}"
    return f"significant_for_{competitor}"


def _focus_pair(
    ranked: list[str], is_ours
) -> tuple[str, str] | None:
    """Best of our methods against the best outside method, for the paper's claims."""
    ours = [m for m in ranked if is_ours(m)]
    theirs = [m for m in ranked if not is_ours(m)]
    if not ours or not theirs:
        return None
    return ours[0], theirs[0]


def ranking_summary(
    summary: pd.DataFrame,
    wide: pd.DataFrame,
    tests: pd.DataFrame,
    validity: pd.DataFrame,
    vtests: pd.DataFrame,
    srg: pd.DataFrame,
    srg_tests: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    def emit(
        *,
        table: str,
        dataset: str,
        metric: str,
        metric_family: str,
        subset_policy: str,
        means: pd.Series,
        ranked: list[str],
        is_ours,
        lookup,
        binary: bool = False,
    ) -> None:
        best, competitor, rule = _pick_competitor(ranked, is_ours)
        rows.append(
            _comparison_row(
                table=table,
                dataset=dataset,
                metric=metric,
                metric_family=metric_family,
                subset_policy=subset_policy,
                means=means,
                ranked=ranked,
                best=best,
                competitor=competitor,
                rule=rule,
                role="primary",
                test=lookup(best, competitor),
                binary=binary,
            )
        )
        focus = _focus_pair(ranked, is_ours)
        if focus and {focus[0], focus[1]} != {best, competitor}:
            ours, theirs = focus
            rows.append(
                _comparison_row(
                    table=table,
                    dataset=dataset,
                    metric=metric,
                    metric_family=metric_family,
                    subset_policy=subset_policy,
                    means=means,
                    ranked=ranked,
                    best=ours,
                    competitor=theirs,
                    rule="best_ours_vs_best_external",
                    role="ours_vs_external",
                    test=lookup(ours, theirs),
                    binary=binary,
                )
            )

    # validity: ranked by flip rate, compared with an exact McNemar test
    for dataset, g in validity.groupby("dataset", observed=True):
        ranked_frame = g.sort_values("validity_rate", ascending=False)
        means = ranked_frame.set_index("method")["validity_rate"]
        emit(
            table="validity",
            dataset=dataset,
            metric="class_flip",
            metric_family="validity",
            subset_policy="attempted",
            means=means,
            ranked=ranked_frame["method"].tolist(),
            is_ours=lambda m: m in CS_METHODS,
            lookup=lambda a, b: _lookup_test(vtests, dataset=dataset, method_a=a, method_b=b),
            binary=True,
        )

    # per-sample metrics ranked on each method's own successes, as the tables do
    own = summary[summary["subset"] == "own_flips"]
    for policy in SUBSET_POLICIES:
        for dataset in order_datasets(own["dataset"].unique()):
            for metric in TESTED_METRICS:
                block = own[(own["dataset"] == dataset) & (own["metric"] == metric)]
                if block.empty:
                    continue
                higher = bool(METRIC_HIGHER_BETTER.get(metric))
                ranked_frame = block.sort_values("mean", ascending=not higher)
                ranked = ranked_frame["method"].tolist()
                if len(ranked) < 2:
                    continue
                means = ranked_frame.set_index("method")["mean"]
                emit(
                    table="per_sample_metric",
                    dataset=dataset,
                    metric=metric,
                    metric_family=METRIC_TO_FAMILY.get(metric, "other"),
                    subset_policy=policy,
                    means=means,
                    ranked=ranked,
                    is_ours=lambda m: m in CS_METHODS,
                    lookup=lambda a, b, _m=metric, _p=policy, _d=dataset: _lookup_test(
                        tests,
                        dataset=_d,
                        metric=_m,
                        subset_policy=_p,
                        method_a=a,
                        method_b=b,
                    ),
                )

    # SRG, with phenotype aggregations standing in for the CounterStroke readouts
    if not srg.empty:
        for dataset, g in srg.groupby("dataset", observed=True):
            means = (
                g.groupby("attribution_method")["srg_score"].mean().sort_values(ascending=False)
            )
            ranked = means.index.tolist()
            if len(ranked) < 2:
                continue
            emit(
                table="srg",
                dataset=dataset,
                metric="srg_score",
                metric_family="srg",
                subset_policy="all_samples",
                means=means,
                ranked=ranked,
                is_ours=lambda m: m in SRG_PHENOTYPE_METHODS,
                lookup=lambda a, b, _d=dataset: _lookup_test(
                    srg_tests, dataset=_d, method_a=a, method_b=b
                ),
            )

    out = pd.DataFrame(rows)
    # Holm across the comparisons each table actually reports
    return _add_holm(
        out,
        ["table", "dataset", "metric_family", "subset_policy", "comparison_role"],
        "p_holm_reported",
    )




def srg_tests_table(srg: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if srg.empty:
        return pd.DataFrame()
    rows = []
    for dataset, g in srg.groupby("dataset", observed=True):
        pivot = g.pivot_table(
            index="sample_idx", columns="attribution_method", values="srg_score"
        )
        methods = [m for m in pivot.columns]
        for first, second in itertools.combinations(methods, 2):
            pair = pivot[[first, second]].dropna()
            if pair.empty:
                continue
            result = paired_test(pair[first].to_numpy(), pair[second].to_numpy(), rng)
            better = None
            if np.isfinite(result["mean_a_tested"]) and np.isfinite(result["mean_b_tested"]):
                better = first if result["mean_a_tested"] > result["mean_b_tested"] else second
            rows.append(
                {
                    "dataset": dataset,
                    "metric": "srg_score",
                    "method_a": first,
                    "method_b": second,
                    "family_a": "phenotype" if first in SRG_PHENOTYPE_METHODS else "saliency",
                    "family_b": "phenotype" if second in SRG_PHENOTYPE_METHODS else "saliency",
                    "identical_scores": bool(result["test"] == "identical"),
                    "better_by_tested_mean": better,
                    **result,
                }
            )
    out = pd.DataFrame(rows)
    return _add_holm(out, ["dataset"], "p_holm_dataset")


def srg_summary_table(srg: pd.DataFrame) -> pd.DataFrame:
    if srg.empty:
        return pd.DataFrame()
    grouped = srg.groupby(["dataset", "attribution_method", "attribution_family"], observed=True)
    out = grouped["srg_score"].agg(["count", "mean", "std", "median"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"])
    out["ci95_low"] = out["mean"] - Z_975 * out["sem"]
    out["ci95_high"] = out["mean"] + Z_975 * out["sem"]
    phen = srg.groupby(["dataset", "attribution_method"], observed=True)["n_phenotypes"]
    counts = phen.agg(["min", "max"]).rename(
        columns={"min": "n_phenotypes_min", "max": "n_phenotypes_max"}
    )
    out = out.merge(counts, on=["dataset", "attribution_method"], how="left")
    out["rank_in_dataset"] = out.groupby("dataset")["mean"].rank(ascending=False, method="min")
    return out.sort_values(["dataset", "rank_in_dataset"])




def set_level_ranking(distribution: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not distribution.empty:
        for (dataset, scope, metric, reference), g in distribution.groupby(
            ["dataset", "scope", "metric", "reference"], observed=True
        ):
            ranked = g.sort_values("value")
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "dataset": dataset,
                        "scope": scope,
                        "metric": metric,
                        "reference": reference,
                        "method": row["method"],
                        "value": row["value"],
                        "value_std": row.get("value_std", np.nan),
                        "rank": rank,
                        "n_set": row.get("n_set", np.nan),
                        "n_class_flip": row.get("n_class_flip", np.nan),
                        "n_failed_cf_in_set": row.get("n_failed_cf_in_set", np.nan),
                        "includes_failed_cf": row.get("includes_failed_cf", False),
                        "tested": False,
                    }
                )

    # per-sample CMMD as the flip-only sensitivity check for the set-level scores
    cmmd = summary[
        summary["metric"].isin(["cmmd_vs_source_y_true_class", "cmmd_vs_predicted_class"])
        & summary["subset"].isin(["own_flips", "available", "common_success"])
    ]
    for (dataset, metric, subset), g in cmmd.groupby(
        ["dataset", "metric", "subset"], observed=True
    ):
        ranked = g.sort_values("mean")
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "scope": f"per_sample_cmmd::{subset}",
                    "metric": metric,
                    "reference": row["subset_definition"],
                    "method": row["method"],
                    "value": row["mean"],
                    "value_std": row["std"],
                    "rank": rank,
                    "n_set": row["n"],
                    "n_class_flip": np.nan,
                    "n_failed_cf_in_set": np.nan,
                    "includes_failed_cf": subset != "own_flips",
                    "tested": True,
                }
            )
    return pd.DataFrame(rows)


def scm_ranking(effects: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not effects.empty:
        for keys, g in effects.groupby(
            ["dataset", "graph", "intervened_parent", "measured_attribute"], observed=True
        ):
            dataset, graph, parent, attribute = keys
            higher = bool(g["higher_is_better"].iloc[0])
            ranked = g.sort_values("mean", ascending=not higher)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "dataset": dataset,
                        "graph": graph,
                        "quantity": f"do({parent}) -> {attribute}",
                        "is_own_attribute": row["is_own_attribute"],
                        "metric_kind": row["metric_kind"],
                        "method": row["method"],
                        "mean": row["mean"],
                        "std": row["std"],
                        "rank": rank,
                        "tested": False,
                    }
                )
    if not models.empty:
        for quantity in ("fid", "minimality_mean"):
            for (dataset, graph), g in models.groupby(["dataset", "graph"], observed=True):
                ranked = g.sort_values(quantity)
                for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                    rows.append(
                        {
                            "dataset": dataset,
                            "graph": graph,
                            "quantity": quantity,
                            "is_own_attribute": False,
                            "metric_kind": "set_level" if quantity == "fid" else "mean_of_samples",
                            "method": row["method"],
                            "mean": row[quantity],
                            "std": row.get("minimality_std", np.nan)
                            if quantity == "minimality_mean"
                            else np.nan,
                            "rank": rank,
                            "tested": False,
                        }
                    )
    return pd.DataFrame(rows)




def diagnostics(
    wide: pd.DataFrame,
    validity: pd.DataFrame,
    summary: pd.DataFrame,
    srg: pd.DataFrame,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    def add(kind: str, dataset: str, detail: str, value: object) -> None:
        rows.append({"kind": kind, "dataset": dataset, "detail": detail, "value": value})

    # how often the three CounterStroke readouts return the same individual
    for dataset, g in wide.groupby("dataset", observed=True):
        pivot = g.pivot_table(index="sample_idx", columns="method", values="l2_distance", observed=True)
        for first, second in itertools.combinations(sorted(CS_METHODS), 2):
            if first not in pivot.columns or second not in pivot.columns:
                continue
            pair = pivot[[first, second]].dropna()
            if pair.empty:
                continue
            identical = float((pair[first] == pair[second]).mean())
            add(
                "readout_collapse",
                dataset,
                f"{first} and {second} select the same individual",
                round(identical, 4),
            )

    # subset sizes that the tables depend on
    for dataset, g in wide.groupby("dataset", observed=True):
        images = g["sample_idx"].nunique()
        common = int(g.drop_duplicates("sample_idx")["in_common_success"].sum())
        add("subset_size", dataset, "images in the manifest", images)
        add("subset_size", dataset, "images flipped by every method", common)

    # embedding margin: positive means the counterfactual moved toward the predicted class
    margins = summary[
        (summary["metric"] == "embedding_margin_src_minus_pred")
        & (summary["subset"] == "own_flips")
    ]
    for _, row in margins.iterrows():
        add(
            "embedding_margin",
            row["dataset"],
            f"{row['method']} mean squared-distance margin (source minus predicted)",
            round(float(row["mean"]), 2),
        )

    # ties that the colouring in the tables cannot distinguish
    for _, row in ranking[ranking["tie_at_3sf"]].iterrows():
        add(
            "display_tie",
            row["dataset"],
            (
                f"{row['metric']} ({row['subset_policy']}, {row['comparison_role']}): "
                f"{row['best']} and {row['competitor']} agree to three significant digits"
            ),
            f"{row['best_mean_displayed']:.5g} vs {row['competitor_mean_displayed']:.5g}",
        )

    # attribution methods that are numerically identical
    if not srg.empty:
        for dataset, g in srg.groupby("dataset", observed=True):
            pivot = g.pivot_table(
                index="sample_idx", columns="attribution_method", values="srg_score"
            )
            for first, second in itertools.combinations(pivot.columns, 2):
                pair = pivot[[first, second]].dropna()
                if pair.empty:
                    continue
                if float((pair[first] - pair[second]).abs().max()) == 0.0:
                    add(
                        "duplicate_attribution",
                        dataset,
                        f"{first} and {second} are identical on every sample",
                        int(len(pair)),
                    )
            phen = g[g["attribution_family"] == "phenotype"]
            if not phen.empty:
                add(
                    "phenotype_population",
                    dataset,
                    "phenotypes per image behind the attribution maps (min-max)",
                    f"{int(phen['n_phenotypes'].min())}-{int(phen['n_phenotypes'].max())}",
                )

    # validity and target attainment side by side, with the comparability caveat
    for _, row in validity.iterrows():
        add(
            "target_attainment",
            row["dataset"],
            (
                f"{row['method']} reached the intended target class "
                f"(targeted method: {bool(row['receives_target_class'])})"
            ),
            f"{int(row['n_target_hit'])}/{int(row['n_attempted'])}",
        )

    return pd.DataFrame(rows)




def _fmt_p(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def stats_report(
    ranking: pd.DataFrame,
    validity: pd.DataFrame,
    srg_summary: pd.DataFrame,
    set_level: pd.DataFrame,
    diag: pd.DataFrame,
    scm: pd.DataFrame,
) -> Path:
    lines = [
        "# Statistics report",
        "",
        "Every cell the paper marks as best, second-best or significant, with the "
        "comparison behind it. Quote `p_holm_reported` in the paper.",
        "",
        PROTOCOL_TEXT.replace("# Statistical protocol", "## Protocol", 1),
        "",
        "## Validity",
        "",
    ]
    for dataset in order_datasets(validity["dataset"].unique()):
        g = validity[validity["dataset"] == dataset].sort_values(
            "validity_rate", ascending=False
        )
        lines.append(f"### {dataset}")
        lines.append("")
        for _, row in g.iterrows():
            lines.append(
                f"- {row['method']}: {int(row['n_class_flip'])}/{int(row['n_attempted'])} "
                f"= {row['validity_rate'] * 100:.0f}% "
                f"(Wilson 95% {row['validity_wilson_low'] * 100:.1f}-"
                f"{row['validity_wilson_high'] * 100:.1f})"
            )
        top = ranking[(ranking["table"] == "validity") & (ranking["dataset"] == dataset)]
        for _, row in top.iterrows():
            lines.append(
                f"- comparison ({row['comparison_role']}): {row['best']} vs "
                f"{row['competitor']} [{row['rule']}], exact McNemar on n={row['n']}, "
                f"p={_fmt_p(row['p_raw'])}, Holm p={_fmt_p(row['p_holm_reported'])}, "
                f"rate difference {row['effect_size']:+.3f}"
            )
        lines.append("")

    lines += ["## Per-sample metrics", ""]
    per_sample = ranking[ranking["table"] == "per_sample_metric"]
    for policy in SUBSET_POLICIES:
        lines.append(f"### Subset policy: {policy}")
        lines.append("")
        lines.append(SUBSET_POLICIES[policy])
        lines.append("")
        for dataset in order_datasets(per_sample["dataset"].unique()):
            block = per_sample[
                (per_sample["dataset"] == dataset) & (per_sample["subset_policy"] == policy)
            ]
            if block.empty:
                continue
            lines.append(f"#### {dataset}")
            lines.append("")
            for _, row in block.iterrows():
                flag = f"  [{row['verdict']}]"
                tie = "  [same to 3 significant digits]" if row["tie_at_3sf"] else ""
                role = "" if row["comparison_role"] == "primary" else f" ({row['comparison_role']})"
                lines.append(
                    f"- {row['metric']}{role}: {row['best']} "
                    f"({row['best_mean_displayed']:.4g} displayed, "
                    f"{row['best_mean_tested']:.4g} on the tested subset) vs "
                    f"{row['competitor']} "
                    f"({row['competitor_mean_displayed']:.4g} / "
                    f"{row['competitor_mean_tested']:.4g}); {row['test']} n={row['n']}, "
                    f"p={_fmt_p(row['p_raw'])}, Holm p={_fmt_p(row['p_holm_reported'])}, "
                    f"{row['effect_size_kind']}={row['effect_size']:+.3f}{tie}{flag}"
                )
            lines.append("")

    if not srg_summary.empty:
        lines += ["## Symmetric Relevance Gain", ""]
        for dataset in order_datasets(srg_summary["dataset"].unique()):
            g = srg_summary[srg_summary["dataset"] == dataset].sort_values("rank_in_dataset")
            lines.append(f"### {dataset}")
            lines.append("")
            for _, row in g.iterrows():
                lines.append(
                    f"- {row['attribution_method']} ({row['attribution_family']}): "
                    f"{row['mean']:.3f} +/- {row['std']:.3f}, n={int(row['count'])}, "
                    f"95% CI [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
                )
            top = ranking[(ranking["table"] == "srg") & (ranking["dataset"] == dataset)]
            for _, row in top.iterrows():
                lines.append(
                    f"- comparison ({row['comparison_role']}): {row['best']} vs "
                    f"{row['competitor']} [{row['rule']}], "
                    f"{row['test']} n={row['n']}, p={_fmt_p(row['p_raw'])}, "
                    f"Holm p={_fmt_p(row['p_holm_reported'])}, "
                    f"{row['effect_size_kind']}={row['effect_size']:+.3f}"
                )
            lines.append("")

    if not set_level.empty:
        lines += [
            "## Set-level distributional scores",
            "",
            "Ranked, not tested. Sets that contain counterfactuals which never "
            "flipped are flagged, because those members stay close to the factual "
            "image and flatter the score.",
            "",
        ]
        contaminated = set_level[
            (set_level["includes_failed_cf"] == True) & (~set_level["tested"])  # noqa: E712
        ]
        for dataset in order_datasets(set_level["dataset"].unique()):
            block = set_level[
                (set_level["dataset"] == dataset)
                & (set_level["scope"] == "paired_vs_factual")
                & (set_level["metric"] == "FID")
            ].sort_values("rank")
            if block.empty:
                continue
            lines.append(f"### {dataset} (paired FID)")
            lines.append("")
            for _, row in block.iterrows():
                note = (
                    f" [{int(row['n_failed_cf_in_set'])} of {int(row['n_set'])} did not flip]"
                    if row["includes_failed_cf"]
                    else ""
                )
                lines.append(f"- {row['method']}: {row['value']:.2f}{note}")
            lines.append("")
        lines.append(
            f"Contaminated set-level entries: {len(contaminated)} of "
            f"{len(set_level[~set_level['tested']])}."
        )
        lines.append("")

    if not scm.empty:
        lines += ["## Deep-SCM", "", "Ranked without tests; only aggregates are available.", ""]
        own = scm[scm["is_own_attribute"]]
        for (dataset, graph), g in own.groupby(["dataset", "graph"], observed=True):
            lines.append(f"### {dataset} ({graph} graph), own-attribute effects")
            lines.append("")
            for quantity, block in g.groupby("quantity", observed=True):
                best = block[block["rank"] == 1]["method"].tolist()
                lines.append(
                    f"- {quantity}: best {', '.join(map(str, best))} "
                    f"({block[block['rank'] == 1]['mean'].iloc[0]:.3f})"
                )
            lines.append("")
        side = scm[(~scm["is_own_attribute"]) & (scm["metric_kind"] != "set_level")]
        lines.append(
            f"Effects on non-intervened attributes available but unreported in the "
            f"paper: {side['quantity'].nunique()} quantities."
        )
        lines.append("")

    if not diag.empty:
        lines += ["## Diagnostics", ""]
        for kind, g in diag.groupby("kind"):
            lines.append(f"### {kind}")
            lines.append("")
            for _, row in g.iterrows():
                lines.append(f"- {row['dataset']}: {row['detail']} = {row['value']}")
            lines.append("")

    return write_text("\n".join(lines), STATS / "STATS_REPORT.md")




def _read(name: str) -> pd.DataFrame:
    path = CONSOLIDATED / name
    if not path.exists():
        LOG.warning("missing consolidated table: %s", name)
        return pd.DataFrame()
    return pd.read_csv(path)


def run(verbose: bool = False) -> dict:
    log = setup_logging(verbose)
    STATS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    wide = _read("per_sample_wide.csv")
    if wide.empty:
        raise SystemExit("run --stage consolidate first: per_sample_wide.csv is missing")
    summary = _read("metric_summary.csv")
    validity_raw = _read("validity.csv")
    distribution = _read("distribution_set_level.csv")
    srg = _read("srg_per_sample.csv")
    scm_effects = _read("scm_effects.csv")
    scm_models = _read("scm_model_level.csv")

    log.info("testing validity with exact McNemar")
    vtests = validity_tests(wide)
    vtable = validity_table(validity_raw)

    log.info("testing %d continuous metrics across all method pairs", len(TESTED_METRICS))
    ptests = paired_tests(wide, rng)

    log.info("testing attribution methods")
    stests = srg_tests_table(srg, rng)
    ssummary = srg_summary_table(srg)

    ranking = ranking_summary(summary, wide, ptests, validity_raw, vtests, srg, stests)
    sets = set_level_ranking(distribution, summary)
    scm = scm_ranking(scm_effects, scm_models)
    diag = diagnostics(wide, validity_raw, summary, srg, ranking)

    tables = {
        "validity_summary.csv": vtable,
        "validity_tests.csv": vtests,
        "paired_tests.csv": ptests,
        "srg_summary.csv": ssummary,
        "srg_tests.csv": stests,
        "ranking_summary.csv": ranking,
        "set_level_ranking.csv": sets,
        "scm_ranking.csv": scm,
        "diagnostics.csv": diag,
    }

    outputs: list[Path] = []
    for name, df in tables.items():
        if df is None or df.empty:
            log.warning("skipping empty table %s", name)
            continue
        outputs.append(write_csv(df, STATS / name))
    outputs.append(write_text(PROTOCOL_TEXT, STATS / "PROTOCOL.md"))
    outputs.append(stats_report(ranking, vtable, ssummary, sets, diag, scm))

    unsupported = (
        int((~ranking["test_agrees_with_ranking"]).sum()) if not ranking.empty else 0
    )
    stage = record_stage(
        "stats",
        outputs,
        extra={
            "n_tables": len(outputs),
            "n_paired_tests": len(ptests),
            "n_validity_tests": len(vtests),
            "n_srg_tests": len(stests),
            "reported_comparisons": len(ranking),
            "reported_comparisons_without_support": unsupported,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "seed": SEED,
        },
    )
    log.info("stats complete: %d files, %d reported comparisons", len(outputs), len(ranking))
    return stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    run(verbose=args.verbose)


if __name__ == "__main__":
    main()

