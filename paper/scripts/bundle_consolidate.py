#!/usr/bin/env python3
"""Stage 2: turn the gathered raw artifacts into canonical tidy tables."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle_common import (  # noqa: E402
    CONSOLIDATED,
    CS_METHODS,
    METHOD_FAMILY,
    METHOD_IS_TARGETED,
    METHOD_LABELS,
    METRIC_HIGHER_BETTER,
    RAW,
    SRG_DATASET_LABELS,
    SRG_METHOD_LABELS,
    SRG_PHENOTYPE_METHODS,
    Z_975,
    canonical_methods,
    order_datasets,
    order_methods,
    record_stage,
    setup_logging,
    wilson_interval,
    write_csv,
    write_text,
)

LOG = setup_logging()

TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz

# per-sample metric columns carried into the tidy table
PER_SAMPLE_METRICS: list[str] = [
    "l2_distance",
    "sparsity_count",
    "sparsity_frac",
    "ssim_to_original",
    "lpips_to_original",
    "ssim_to_random_pred_class",
    "lpips_to_random_pred_class",
    "p_true_orig",
    "p_true_cf",
    "delta_p_true",
    "p_intended_target_cf",
    "p_pred_cf",
    "inception_sqdist_to_source_class_mean",
    "inception_sqdist_to_predicted_class_mean",
    "embedding_margin_src_minus_pred",
    "cmmd_vs_source_y_true_class",
    "cmmd_vs_predicted_class",
]

SUBSET_DEFINITIONS: dict[str, str] = {
    "own_flips": "rows where this method changed the predicted class",
    "common_success": "rows where every method changed the predicted class",
    "available": "rows where this method returned a counterfactual, flipped or not",
    "attempted": "all evaluation images, no conditioning",
}

# What the paper's configuration table reports, for cross-checking against the
# checked-in configs (Table 9 in counterstroke_tmlr.tex).
PAPER_CONFIG: dict[str, dict[str, int]] = {
    "MNIST": {"population_size": 1000, "max_generations": 500},
    "CIFAR-10": {"population_size": 1000, "max_generations": 500},
    "ImageNet-100": {"population_size": 2000, "max_generations": 500},
}

CONFIG_DIRS: dict[str, str] = {
    "MNIST": "MNIST",
    "CIFAR-10": "Cifar10",
    "ImageNet-100": "Imagenet",
}




def _read_raw_csv(relative: str, silent: bool = False) -> pd.DataFrame | None:
    path = RAW / relative
    if not path.exists():
        if not silent:
            LOG.warning("missing raw input: %s", relative)
        return None
    return pd.read_csv(path)


def _read_main_or_per_dataset(main_name: str, per_dataset_name: str) -> pd.DataFrame | None:
    """Try the pre-aggregated raw/main/<main_name> first (dev machine); fall back"""
    df = _read_raw_csv(f"main/{main_name}", silent=True)
    if df is not None:
        return df
    df = _read_per_dataset_csv(per_dataset_name)
    if df is None:
        LOG.warning("no source found for %s / %s", main_name, per_dataset_name)
    return df


# Dataset name as it appears in the CSV ``dataset`` column, keyed by the
# folder name used in {Dataset}/outputs/generation/ on the cluster.
_DATASET_FOLDER_TO_LABEL: dict[str, str] = {
    "MNIST": "MNIST",
    "Cifar10": "CIFAR-10",
    "Imagenet": "ImageNet-100",
}


def _read_per_dataset_csv(filename: str) -> pd.DataFrame | None:
    """Combine per-dataset CSVs from raw/generation when the aggregated file"""
    gen_root = RAW / "generation"
    if not gen_root.exists():
        return None
    frames = []
    for folder, label in _DATASET_FOLDER_TO_LABEL.items():
        path = gen_root / folder / filename
        if not path.exists():
            LOG.debug("per-dataset file absent: %s", path.relative_to(RAW))
            continue
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            df.insert(0, "dataset", label)
        frames.append(df)
    if not frames:
        return None
    LOG.info("combined %d per-dataset %s files", len(frames), filename)
    return pd.concat(frames, ignore_index=True)


def method_table() -> pd.DataFrame:
    rows = []
    for raw, label in METHOD_LABELS.items():
        rows.append(
            {
                "raw_name": raw,
                "label": label,
                "family": METHOD_FAMILY.get(label, "unknown"),
                "is_counterstroke": label in CS_METHODS,
                "receives_target_class": METHOD_IS_TARGETED.get(label),
            }
        )
    return pd.DataFrame(rows)


def _derive_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive flag columns that the post-processing script adds but the raw"""
    df = df.copy()
    if "available" not in df.columns:
        df["available"] = df["has_valid_cf"].fillna(0).astype(int)
    if "class_flip" not in df.columns:
        pred = pd.to_numeric(df["pred_class_cf"], errors="coerce")
        flipped = (df["has_valid_cf"].fillna(0).astype(int) == 1) & (pred == pred) & (
            pred.astype("Int64") != df["y_true"].astype("Int64")
        )
        df["class_flip"] = flipped.astype(int)
    if "target_hit" not in df.columns:
        pred = pd.to_numeric(df["pred_class_cf"], errors="coerce")
        intended = pd.to_numeric(df.get("y_intended_target", pd.Series(dtype=float)), errors="coerce")
        hit = (df["has_valid_cf"].fillna(0).astype(int) == 1) & (pred == pred) & (
            pred.astype("Int64") == intended.astype("Int64")
        )
        df["target_hit"] = hit.astype(int)
    return df


def build_wide(report: list[str]) -> pd.DataFrame:
    """Canonical wide per-sample table with derived metrics and subset flags."""
    df = _read_main_or_per_dataset(
        "main_per_sample_all_datasets.csv", "per_sample_metrics.csv"
    )
    if df is None:
        raise SystemExit(
            "cannot consolidate: neither raw/main/main_per_sample_all_datasets.csv "
            "nor raw/generation/{MNIST,Cifar10,Imagenet}/per_sample_metrics.csv found"
        )
    df = _derive_missing_columns(df)
    df = canonical_methods(df)

    cmmd = _read_main_or_per_dataset("main_cmmd_per_sample.csv", "cmmd_per_sample.csv")
    if cmmd is not None:
        cmmd = canonical_methods(cmmd)
        keep = [
            "dataset",
            "sample_idx",
            "method",
            "cmmd_vs_source_y_true_class",
            "cmmd_vs_predicted_class",
        ]
        df = df.merge(cmmd[keep], on=["dataset", "sample_idx", "method"], how="left")
    else:
        df["cmmd_vs_source_y_true_class"] = np.nan
        df["cmmd_vs_predicted_class"] = np.nan

    df["embedding_margin_src_minus_pred"] = (
        df["inception_sqdist_to_source_class_mean"]
        - df["inception_sqdist_to_predicted_class_mean"]
    )

    # images every method flipped: the clean subset for cross-method comparison
    flips = df.pivot_table(
        index=["dataset", "sample_idx"], columns="method", values="class_flip", aggfunc="max"
    )
    common = flips.fillna(0).min(axis=1).rename("in_common_success").reset_index()
    df = df.merge(common, on=["dataset", "sample_idx"], how="left")
    df["in_common_success"] = df["in_common_success"].fillna(0).astype(int)

    df["method_family"] = df["method"].map(METHOD_FAMILY)
    df["receives_target_class"] = df["method"].map(METHOD_IS_TARGETED)

    for dataset, group in df.groupby("dataset"):
        n_common = int(group.drop_duplicates("sample_idx")["in_common_success"].sum())
        n_images = group["sample_idx"].nunique()
        report.append(
            f"- {dataset}: {n_common}/{n_images} images flipped by every method "
            f"(common-success subset)"
        )

    ordered_cols = [
        "dataset",
        "sample_idx",
        "y_true",
        "y_intended_target",
        "pred_class_cf",
        "method",
        "method_family",
        "receives_target_class",
        "has_valid_cf",
        "available",
        "class_flip",
        "target_hit",
        "in_common_success",
        *PER_SAMPLE_METRICS,
    ]
    present = [c for c in ordered_cols if c in df.columns]
    df = df[present].copy()
    df["dataset"] = pd.Categorical(df["dataset"], order_datasets(df["dataset"]))
    df["method"] = pd.Categorical(df["method"], order_methods(df["method"]))
    return df.sort_values(["dataset", "method", "sample_idx"]).reset_index(drop=True)


def build_tidy(wide: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "dataset",
        "sample_idx",
        "y_true",
        "y_intended_target",
        "pred_class_cf",
        "method",
        "method_family",
        "receives_target_class",
        "available",
        "class_flip",
        "target_hit",
        "in_common_success",
    ]
    value_cols = [c for c in PER_SAMPLE_METRICS if c in wide.columns]
    tidy = wide.melt(
        id_vars=[c for c in id_cols if c in wide.columns],
        value_vars=value_cols,
        var_name="metric",
        value_name="value",
    )
    tidy["higher_is_better"] = tidy["metric"].map(METRIC_HIGHER_BETTER)
    return tidy.sort_values(["dataset", "metric", "method", "sample_idx"]).reset_index(drop=True)


def build_validity(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), g in wide.groupby(["dataset", "method"], observed=True):
        n = len(g)
        n_available = int(g["available"].sum())
        n_flip = int(g["class_flip"].sum())
        n_target = int(g["target_hit"].sum()) if "target_hit" in g else 0
        lo, hi = wilson_interval(n_flip, n)
        alo, ahi = wilson_interval(n_available, n)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "receives_target_class": METHOD_IS_TARGETED.get(method),
                "n_attempted": n,
                "n_available": n_available,
                "availability_rate": n_available / n if n else np.nan,
                "availability_wilson_low": alo,
                "availability_wilson_high": ahi,
                "n_class_flip": n_flip,
                "validity_rate": n_flip / n if n else np.nan,
                "validity_wilson_low": lo,
                "validity_wilson_high": hi,
                "n_target_hit": n_target,
                "target_attainment_rate": n_target / n if n else np.nan,
                "target_attainment_comparable": False,
                "notes": (
                    "target attainment is not comparable across methods: "
                    "CounterStroke and GradCF are untargeted"
                ),
            }
        )
    return pd.DataFrame(rows)


def _subset_mask(g: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "own_flips":
        return g["class_flip"] == 1
    if subset == "common_success":
        return g["in_common_success"] == 1
    if subset == "available":
        return g["available"] == 1
    return pd.Series(True, index=g.index)


def build_metric_summary(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [c for c in PER_SAMPLE_METRICS if c in wide.columns]
    for (dataset, method), g in wide.groupby(["dataset", "method"], observed=True):
        for subset in SUBSET_DEFINITIONS:
            sub = g[_subset_mask(g, subset)]
            for metric in metrics:
                values = pd.to_numeric(sub[metric], errors="coerce").dropna()
                n = int(values.size)
                mean = float(values.mean()) if n else np.nan
                std = float(values.std(ddof=1)) if n > 1 else np.nan
                sem = std / np.sqrt(n) if n > 1 and np.isfinite(std) else np.nan
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "higher_is_better": METRIC_HIGHER_BETTER.get(metric),
                        "subset": subset,
                        "subset_definition": SUBSET_DEFINITIONS[subset],
                        "n": n,
                        "mean": mean,
                        "std": std,
                        "median": float(values.median()) if n else np.nan,
                        "q25": float(values.quantile(0.25)) if n else np.nan,
                        "q75": float(values.quantile(0.75)) if n else np.nan,
                        "sem": sem,
                        "ci95_low": mean - Z_975 * sem if np.isfinite(sem) else np.nan,
                        "ci95_high": mean + Z_975 * sem if np.isfinite(sem) else np.nan,
                    }
                )
    return pd.DataFrame(rows)




def build_distribution(validity: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    """Merge paired, batch and stratified set-level scores into one long table."""
    flips = validity.set_index(["dataset", "method"])["n_class_flip"].to_dict()
    available = validity.set_index(["dataset", "method"])["n_available"].to_dict()

    records: list[dict] = []

    paired = _read_main_or_per_dataset(
        "main_distribution_metrics.csv", "distribution_metrics.csv"
    )
    if paired is not None:
        paired = canonical_methods(paired)
        spec = [
            ("FID", "frechet_inception_distance_cf_vs_originals", None, "paired factual images"),
            ("KID", "kid_mean_cf_vs_originals", "kid_std_cf_vs_originals", "paired factual images"),
            ("CMMD", "cmmd_orig_vs_cf_distribution", None, "paired factual images"),
        ]
        for _, row in paired.iterrows():
            for metric, col, std_col, reference in spec:
                records.append(
                    {
                        "dataset": row["dataset"],
                        "method": row["method"],
                        "scope": "paired_vs_factual",
                        "metric": metric,
                        "reference": reference,
                        "value": float(row[col]),
                        "value_std": float(row[std_col]) if std_col else np.nan,
                        "n_set": int(row.get("n_paired", np.nan)),
                    }
                )

    batch = _read_main_or_per_dataset("main_batch_fid.csv", "batch_fid.csv")
    if batch is not None:
        batch = canonical_methods(batch)
        spec = [
            (
                "FID",
                "frechet_inception_distance_vs_source_y_true",
                "real images of the factual class",
            ),
            (
                "FID",
                "frechet_inception_distance_vs_predicted_class",
                "real images of the predicted class",
            ),
        ]
        for _, row in batch.iterrows():
            for metric, col, reference in spec:
                records.append(
                    {
                        "dataset": row["dataset"],
                        "method": row["method"],
                        "scope": "batch_vs_class_pool",
                        "metric": metric,
                        "reference": reference,
                        "value": float(row[col]),
                        "value_std": np.nan,
                        "n_set": int(available.get((row["dataset"], row["method"]), np.nan)),
                    }
                )

    strat = _read_main_or_per_dataset(
        "main_cmmd_batch_stratified.csv", "cmmd_batch_stratified.csv"
    )
    if strat is not None:
        strat = canonical_methods(strat)
        spec = [
            ("CMMD", "cmmd_vs_stratified_source_y_true", "real images of the factual class"),
            ("CMMD", "cmmd_vs_stratified_predicted_class", "real images of the predicted class"),
        ]
        for _, row in strat.iterrows():
            for metric, col, reference in spec:
                records.append(
                    {
                        "dataset": row["dataset"],
                        "method": row["method"],
                        "scope": "stratified_vs_class_pool",
                        "metric": metric,
                        "reference": reference,
                        "value": float(row[col]),
                        "value_std": np.nan,
                        "n_set": int(row.get("n_paired", np.nan)),
                    }
                )

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out["n_class_flip"] = [
        flips.get((d, m), np.nan) for d, m in zip(out["dataset"], out["method"])
    ]
    out["n_failed_cf_in_set"] = out["n_set"] - out["n_class_flip"]
    out["includes_failed_cf"] = out["n_failed_cf_in_set"] > 0
    out["set_definition"] = np.where(
        out["includes_failed_cf"],
        "all returned counterfactuals, including those that did not flip",
        "all returned counterfactuals (all of which flipped)",
    )
    out["higher_is_better"] = False

    contaminated = out[out["includes_failed_cf"]][
        ["dataset", "method", "n_set", "n_class_flip"]
    ].drop_duplicates()
    for _, row in contaminated.iterrows():
        report.append(
            f"- set-level scores for {row['method']} on {row['dataset']} cover "
            f"{int(row['n_set'])} counterfactuals of which only "
            f"{int(row['n_class_flip'])} flipped"
        )

    out["dataset"] = pd.Categorical(out["dataset"], order_datasets(out["dataset"]))
    out["method"] = pd.Categorical(out["method"], order_methods(out["method"]))
    return out.sort_values(["dataset", "scope", "metric", "reference", "method"]).reset_index(
        drop=True
    )


def build_cmmd_per_sample(wide: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "sample_idx",
        "y_true",
        "pred_class_cf",
        "method",
        "available",
        "class_flip",
        "in_common_success",
        "cmmd_vs_source_y_true_class",
        "cmmd_vs_predicted_class",
    ]
    return wide[[c for c in cols if c in wide.columns]].copy()


def build_srg_per_sample(report: list[str]) -> pd.DataFrame:
    frames = []
    for slug, label in SRG_DATASET_LABELS.items():
        path = RAW / "explainability" / f"{slug}_srg_summary.csv"
        if not path.exists():
            LOG.debug("pre-built SRG summary absent for %s; will use npz curves", slug)
            continue
        df = pd.read_csv(path)
        df["dataset"] = label
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["attribution_method"] = out["method"].map(lambda m: SRG_METHOD_LABELS.get(m, m))
    out["attribution_family"] = np.where(
        out["attribution_method"].isin(SRG_PHENOTYPE_METHODS), "phenotype", "saliency"
    )
    out = out.rename(columns={"method": "raw_method"})
    keep = [
        "dataset",
        "sample_idx",
        "y_true",
        "explanation_class",
        "n_phenotypes",
        "attribution_method",
        "attribution_family",
        "raw_method",
        "srg_score",
    ]
    out = out[[c for c in keep if c in out.columns]]

    for dataset, g in out.groupby("dataset"):
        counts = g.groupby("attribution_method")["sample_idx"].nunique()
        phen = g[g["attribution_family"] == "phenotype"]
        report.append(
            f"- {dataset}: SRG over {int(counts.max())} samples; phenotype maps "
            f"built from {int(phen['n_phenotypes'].min())}-"
            f"{int(phen['n_phenotypes'].max())} phenotypes per image"
        )
    return out


def build_srg_curves(report: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute per-sample AUC(MIF)/AUC(LIF) and mean curves from the npz files."""
    root = RAW / "srg_curves"
    if not root.exists():
        LOG.warning("no SRG curve files gathered; skipping AUC recomputation")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rows: list[dict] = []
    curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    strategies: set[str] = set()

    for path in sorted(root.rglob("*.npz")):
        parts = path.relative_to(root).parts
        slug = parts[0]
        dataset = SRG_DATASET_LABELS.get(slug, slug)
        sample = next((p for p in parts if p.startswith("sample_")), "")
        sample_idx = int(sample.split("_")[-1]) if sample else -1
        raw_method = path.stem
        method = SRG_METHOD_LABELS.get(raw_method, raw_method)

        with np.load(path, allow_pickle=True) as data:
            k = np.asarray(data["k_fractions"], dtype=np.float64)
            mif = np.asarray(data["mif_curve"], dtype=np.float64)
            lif = np.asarray(data["lif_curve"], dtype=np.float64)
            srg = float(data["srg_score"])
            n_pixels = int(data["n_pixels"])
            p_baseline = float(data["p_baseline"])
            prob_class = int(data["prob_class"])
            strategy = str(data["occlusion_strategy"])

        length = min(len(k), len(mif), len(lif))
        k, mif, lif = k[:length], mif[:length], lif[:length]
        span = float(k[-1] - k[0]) or 1.0
        auc_mif = float(TRAPEZOID(mif, k)) / span
        auc_lif = float(TRAPEZOID(lif, k)) / span
        strategies.add(strategy)

        rows.append(
            {
                "dataset": dataset,
                "sample_idx": sample_idx,
                "attribution_method": method,
                "attribution_family": (
                    "phenotype" if method in SRG_PHENOTYPE_METHODS else "saliency"
                ),
                "raw_method": raw_method,
                "srg_score": srg,
                "auc_mif": auc_mif,
                "auc_lif": auc_lif,
                "auc_gap_lif_minus_mif": auc_lif - auc_mif,
                "n_points": length,
                "n_pixels": n_pixels,
                "p_baseline": p_baseline,
                "prob_class": prob_class,
                "occlusion_strategy": strategy,
            }
        )
        curves.setdefault((dataset, method), []).append((k, mif, lif))

    per_sample = pd.DataFrame(rows)
    if per_sample.empty:
        return per_sample, pd.DataFrame(), pd.DataFrame()

    mean_rows: list[dict] = []
    summary_rows: list[dict] = []
    for (dataset, method), items in curves.items():
        length = min(len(k) for k, _, _ in items)
        k_ref = items[0][0][:length]
        mif = np.mean(np.stack([m[:length] for _, m, _ in items]), axis=0)
        lif = np.mean(np.stack([l[:length] for _, _, l in items]), axis=0)
        for i in range(length):
            mean_rows.append(
                {
                    "dataset": dataset,
                    "attribution_method": method,
                    "k_fraction": float(k_ref[i]),
                    "mif_mean": float(mif[i]),
                    "lif_mean": float(lif[i]),
                    "n_samples": len(items),
                }
            )
        span = float(k_ref[-1] - k_ref[0]) or 1.0
        sub = per_sample[
            (per_sample["dataset"] == dataset) & (per_sample["attribution_method"] == method)
        ]
        summary_rows.append(
            {
                "dataset": dataset,
                "attribution_method": method,
                "attribution_family": (
                    "phenotype" if method in SRG_PHENOTYPE_METHODS else "saliency"
                ),
                "n_samples": len(items),
                "srg_mean": float(sub["srg_score"].mean()),
                "srg_std": float(sub["srg_score"].std(ddof=1)) if len(sub) > 1 else np.nan,
                "auc_mif_of_mean_curve": float(TRAPEZOID(mif, k_ref)) / span,
                "auc_lif_of_mean_curve": float(TRAPEZOID(lif, k_ref)) / span,
                "auc_mif_mean_of_samples": float(sub["auc_mif"].mean()),
                "auc_lif_mean_of_samples": float(sub["auc_lif"].mean()),
                "auc_gap_of_mean_curve": (
                    float(TRAPEZOID(lif, k_ref)) / span - float(TRAPEZOID(mif, k_ref)) / span
                ),
            }
        )

    report.append(
        "- SRG corruption strategy recorded in the curve files: "
        + ", ".join(sorted(strategies))
    )
    mean_curves = pd.DataFrame(mean_rows).sort_values(
        ["dataset", "attribution_method", "k_fraction"]
    )
    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "attribution_method"])

    gap = (summary["srg_mean"] - summary["auc_gap_of_mean_curve"]).abs().max()
    report.append(
        f"- SRG equals AUC(LIF) minus AUC(MIF) to {gap:.2e}, so the AUC table is "
        "a decomposition of SRG rather than independent evidence"
    )
    return per_sample, mean_curves, summary


def build_summary_table_crosscheck(
    metric_summary: pd.DataFrame, report: list[str]
) -> pd.DataFrame:
    """Read per-dataset summary_tables.csv files and compare their means to the"""
    gen_root = RAW / "generation"
    frames = []
    for folder, label in _DATASET_FOLDER_TO_LABEL.items():
        path = gen_root / folder / "summary_tables.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["dataset"] = label
        frames.append(df)
    if not frames:
        report.append("- summary_tables.csv: not found for any dataset; skipping cross-check")
        return pd.DataFrame()

    src = pd.concat(frames, ignore_index=True)
    src["method"] = canonical_methods(src[["method"]].rename(columns={"method": "method"}))["method"] \
        if "method" in src.columns else src.get("method", pd.Series(dtype=str))

    # only compare the 'mean' aggregate rows
    means = src[src["aggregate"] == "mean"] if "aggregate" in src.columns else src.copy()
    consolidated = metric_summary[metric_summary["subset"] == "own_flips"]

    rows = []
    for _, row in means.iterrows():
        dataset = row.get("dataset")
        method = row.get("method")
        metric = row.get("metric")
        src_val = float(row.get("value", float("nan")))

        match = consolidated[
            (consolidated["dataset"] == dataset)
            & (consolidated["method"] == method)
            & (consolidated["metric"] == metric)
        ]
        if match.empty or not np.isfinite(src_val):
            continue
        cons_val = float(match["mean"].iloc[0])
        abs_diff = abs(src_val - cons_val)
        rel_diff = abs_diff / abs(cons_val) if cons_val != 0 else float("inf")
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "summary_table_mean": src_val,
                "consolidated_mean": cons_val,
                "abs_diff": abs_diff,
                "rel_diff_pct": rel_diff * 100,
                "agrees": rel_diff < 0.01,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        report.append("- summary_tables.csv: present but no overlapping rows found for cross-check")
        return out

    mismatches = out[~out["agrees"]]
    report.append(
        f"- summary_tables.csv cross-check: {len(out)} overlapping cells, "
        f"{len(mismatches)} disagree by more than 1%"
    )
    for _, row in mismatches.iterrows():
        report.append(
            f"  - {row['dataset']} / {row['method']} / {row['metric']}: "
            f"summary={row['summary_table_mean']:.4g}, "
            f"consolidated={row['consolidated_mean']:.4g} "
            f"({row['rel_diff_pct']:.1f}%)"
        )
    return out


def build_search_config(report: list[str]) -> pd.DataFrame:
    rows = []
    for dataset, folder in CONFIG_DIRS.items():
        path = RAW / "config" / folder / "config.json"
        if not path.exists():
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        evolution = cfg.get("evolution", {}) or {}
        runtime = cfg.get("runtime", {}) or {}
        population = evolution.get("population_size")
        generations = evolution.get("max_generations")
        paper = PAPER_CONFIG.get(dataset, {})
        rows.append(
            {
                "dataset": dataset,
                "source": f"raw/config/{folder}/config.json",
                "population_size": population,
                "max_generations": generations,
                "n_runs": evolution.get("n_runs"),
                "seed": runtime.get("seed"),
                "config_evaluations_per_image": (
                    population * generations if population and generations else np.nan
                ),
                "paper_population_size": paper.get("population_size"),
                "paper_max_generations": paper.get("max_generations"),
                "paper_evaluations_per_image": (
                    paper.get("population_size", 0) * paper.get("max_generations", 0) or np.nan
                ),
                "matches_paper": (
                    population == paper.get("population_size")
                    and generations == paper.get("max_generations")
                ),
                "note": (
                    "evaluations per image are an upper bound derived from the "
                    "configuration, not a measurement"
                ),
            }
        )
    out = pd.DataFrame(rows)
    for _, row in out[~out["matches_paper"].fillna(False)].iterrows():
        report.append(
            f"- {row['dataset']}: checked-in config is "
            f"{row['population_size']}x{row['max_generations']} but the paper's "
            f"configuration table reports "
            f"{row['paper_population_size']}x{row['paper_max_generations']}"
        )
    return out



SCM_METHOD_LABELS: dict[str, str] = {
    "graco_base_o2": "CS_sparse",
    "graco_base_o3": "CS_prox",
    "graco50": "CS_50",
    "vae": "VAE",
    "hvae": "HVAE",
    "gan": "GAN",
}

# continuous MorphoMNIST attributes are reported as absolute errors, the rest as
# attribute-predictor accuracies
SCM_CONTINUOUS_ATTRIBUTES = {"thickness", "intensity"}

_NP_SCALAR = re.compile(r"np\.float(?:32|64)\(([^)]*)\)")


def _parse_pair(value: object) -> tuple[float, float]:
    """Parse ``"[mean, std]"`` into a tuple."""
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return float("nan"), float("nan")
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return float(parsed[0]), float(parsed[1])
    return float("nan"), float("nan")


def _parse_effect_dict(value: object) -> dict[str, tuple[float, float]]:
    """Parse the SCM ``effectiveness_do_X`` dict repr, tolerating np scalars."""
    if not isinstance(value, str) or "{" not in value:
        return {}
    cleaned = _NP_SCALAR.sub(r"\1", value)
    try:
        parsed = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return {}
    out: dict[str, tuple[float, float]] = {}
    if isinstance(parsed, dict):
        for key, item in parsed.items():
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out[str(key)] = (float(item[0]), float(item[1]))
            else:
                try:
                    out[str(key)] = (float(item), float("nan"))
                except (TypeError, ValueError):
                    continue
    return out


def _parse_fid(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return float("nan")


def _parse_minimality(value: object) -> tuple[float, float]:
    if isinstance(value, str) and value.strip().startswith("["):
        return _parse_pair(value)
    if isinstance(value, str):
        mean = re.search(r"mean\s+(-?\d+(?:\.\d+)?)", value)
        std = re.search(r"std\s+(-?\d+(?:\.\d+)?)", value)
        return (
            float(mean.group(1)) if mean else float("nan"),
            float(std.group(1)) if std else float("nan"),
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), float("nan")
    return float("nan"), float("nan")


def _scm_label(row: pd.Series) -> str:
    variant = row.get("variant")
    if isinstance(variant, str) and variant in SCM_METHOD_LABELS:
        return SCM_METHOD_LABELS[variant]
    method = str(row.get("method", "")).lower()
    return SCM_METHOD_LABELS.get(method, method.upper() or "unknown")


def build_scm(report: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects: list[dict] = []
    model_level: list[dict] = []
    unparsed: list[str] = []

    # Try several candidate names: pre-aggregated dev-machine names first, then
    # cluster names produced by benchmark/shared/aggregate_results.py.
    def _bench_path(candidates: list[str]) -> Path | None:
        for name in candidates:
            p = RAW / "benchmark" / name
            if p.exists():
                return p
        return None

    files = {
        "MorphoMNIST": _bench_path(
            ["morphomnist_summary.csv", "summary.csv"]
        ),
        "CelebA": _bench_path(
            ["celeba_summary.csv", "summary.csv"]
        ),
    }
    files = {k: v for k, v in files.items() if v is not None}

    for dataset, path in files.items():
        if not path.exists():
            LOG.warning("missing benchmark summary: %s", path.name)
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            label = _scm_label(row)
            graph = row.get("graph", "default")
            fid = _parse_fid(row.get("fid"))
            min_mean, min_std = _parse_minimality(row.get("minimality"))
            model_level.append(
                {
                    "dataset": dataset,
                    "graph": graph,
                    "method": label,
                    "variant": row.get("variant"),
                    "is_counterstroke": label in CS_METHODS,
                    "fid": fid,
                    "minimality_mean": min_mean,
                    "minimality_std": min_std,
                }
            )

            # CounterStroke rows: eff_do_<parent>_<attribute> = "[mean, std]"
            for column in df.columns:
                if not column.startswith("eff_do_"):
                    continue
                value = row.get(column)
                if not isinstance(value, str) or not value.strip().startswith("["):
                    continue
                mean, std = _parse_pair(value)
                if not np.isfinite(mean):
                    unparsed.append(f"{dataset}/{label}/{column}: {value!r}")
                    continue
                body = column[len("eff_do_") :]
                parent, attribute = _split_parent_attribute(body)
                effects.append(
                    _effect_row(dataset, graph, label, row, parent, attribute, mean, std)
                )

            # SCM baselines: effectiveness_do_<parent> = dict of attribute -> (mean, std)
            for column in df.columns:
                if not column.startswith("effectiveness_do_"):
                    continue
                parsed = _parse_effect_dict(row.get(column))
                if not parsed:
                    if isinstance(row.get(column), str) and row.get(column).strip():
                        unparsed.append(f"{dataset}/{label}/{column}")
                    continue
                parent = column[len("effectiveness_do_") :]
                for attribute, (mean, std) in parsed.items():
                    effects.append(
                        _effect_row(dataset, graph, label, row, parent, attribute, mean, std)
                    )

    effects_df = pd.DataFrame(effects)
    model_df = pd.DataFrame(model_level)

    if not effects_df.empty:
        effects_df = effects_df.sort_values(
            ["dataset", "graph", "intervened_parent", "measured_attribute", "method"]
        ).reset_index(drop=True)
        own = effects_df[effects_df["is_own_attribute"]]
        report.append(
            f"- Deep-SCM effect matrix: {len(effects_df)} entries "
            f"({len(own)} own-attribute, {len(effects_df) - len(own)} on "
            f"non-intervened attributes)"
        )
    if unparsed:
        report.append(f"- unparsed Deep-SCM fields: {len(unparsed)}")
        for item in unparsed[:10]:
            report.append(f"  - {item}")
    return effects_df, model_df


def _split_parent_attribute(body: str) -> tuple[str, str]:
    """Split ``Bald_No_Beard`` into (``Bald``, ``No_Beard``) using known names."""
    known = [
        "No_Beard",
        "Eyeglasses",
        "Smiling",
        "Young",
        "Male",
        "Bald",
        "thickness",
        "intensity",
        "digit",
    ]
    for name in known:
        suffix = f"_{name}"
        if body.endswith(suffix) and len(body) > len(suffix):
            return body[: -len(suffix)], name
    parent, _, attribute = body.partition("_")
    return parent, attribute


def _effect_row(
    dataset: str,
    graph: object,
    label: str,
    row: pd.Series,
    parent: str,
    attribute: str,
    mean: float,
    std: float,
) -> dict:
    kind = "abs_error" if attribute in SCM_CONTINUOUS_ATTRIBUTES else "accuracy"
    return {
        "dataset": dataset,
        "graph": graph,
        "method": label,
        "variant": row.get("variant"),
        "is_counterstroke": label in CS_METHODS,
        "intervened_parent": parent,
        "measured_attribute": attribute,
        "is_own_attribute": parent.lower() == attribute.lower(),
        "metric_kind": kind,
        "higher_is_better": kind == "accuracy",
        "mean": mean,
        "std": std,
    }




def paper_crosscheck(
    validity: pd.DataFrame,
    summary: pd.DataFrame,
    srg: pd.DataFrame,
    srg_curves: pd.DataFrame,
    srg_curve_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Structural checks against claims currently printed in the paper."""
    checks: list[dict] = []

    def add(claim: str, paper: str, data: str, ok: bool, where: str) -> None:
        checks.append(
            {"claim": claim, "paper_says": paper, "data_says": data, "agrees": ok, "where": where}
        )

    cs_validity = validity[validity["method"].isin(CS_METHODS)]
    for dataset, g in cs_validity.groupby("dataset"):
        rate = g["validity_rate"].unique()
        flips = sorted(g["n_class_flip"].unique())
        agrees = bool(np.allclose(rate, 1.0))
        add(
            f"CounterStroke validity on {dataset}",
            "100% (perfect validity on all three datasets)",
            f"{flips[0]}/{int(g['n_attempted'].iloc[0])} for every readout",
            agrees,
            "Section 5.1 and Tables 10-12",
        )

    best_val = validity.loc[validity.groupby("dataset")["validity_rate"].idxmax()]
    for _, row in best_val.iterrows():
        top = validity[
            (validity["dataset"] == row["dataset"])
            & (validity["validity_rate"] == row["validity_rate"])
        ]["method"].tolist()
        add(
            f"highest validity on {row['dataset']}",
            "CounterStroke",
            f"{', '.join(map(str, top))} at {row['validity_rate']:.2f}",
            bool(set(top) & CS_METHODS),
            "Tables 10-12 (green cell)",
        )

    own = summary[summary["subset"] == "own_flips"]
    for dataset in own["dataset"].unique():
        sub = own[(own["dataset"] == dataset) & (own["metric"] == "sparsity_count")]
        cs = sub[sub["method"].isin(CS_METHODS)]
        if cs.empty:
            continue
        value = cs["mean"].min()
        add(
            f"CounterStroke edited-pixel count on {dataset}",
            "sparse, human-readable drawing evidence",
            f"{value:.1f} changed values per image (sparsest readout)",
            True,
            "Section 5.2",
        )

    if not srg.empty:
        for dataset, g in srg.groupby("dataset"):
            means = g.groupby("attribution_method")["srg_score"].mean()
            phen = g[g["attribution_family"] == "phenotype"]
            phen_means = phen.groupby("attribution_method")["srg_score"].mean()
            if phen_means.empty:
                continue
            best_phen = phen_means.idxmax()
            add(
                f"best phenotype attribution on {dataset}",
                "spatial coherence",
                f"{best_phen} at {phen_means.max():.3f}",
                best_phen == "Spatial coherence",
                "Section 5.7",
            )
            add(
                f"best attribution overall on {dataset}",
                "phenotype maps lead on MNIST and CIFAR-10",
                f"{means.idxmax()} at {means.max():.3f}",
                True,
                "Section 5.7",
            )

        dupes = _duplicate_attribution_methods(srg)
        add(
            "change frequency and co-change are distinct aggregations",
            "listed as two separate methods with identical values",
            (
                "identical on every sample in " + ", ".join(dupes)
                if dupes
                else "values differ"
            ),
            not dupes,
            "Table 13 and Table 14",
        )

    if not srg_curves.empty:
        strategies = sorted(srg_curves["occlusion_strategy"].unique())
        add(
            "SRG corruption strategy",
            "mean-intensity replacement in normalized image space",
            ", ".join(strategies),
            "mean" in " ".join(strategies).lower(),
            "Section 4.3",
        )

    if not srg_curve_summary.empty:
        delta = (
            srg_curve_summary["srg_mean"] - srg_curve_summary["auc_gap_of_mean_curve"]
        ).abs().max()
        add(
            "AUC(MIF)/AUC(LIF) add information beyond SRG",
            "AUC diagnoses curve shape but does not replace SRG",
            f"SRG == AUC(LIF) - AUC(MIF) to {delta:.2e} for every method",
            False,
            "Section 5.7 and Table 14",
        )

    return pd.DataFrame(checks)


def _duplicate_attribution_methods(srg: pd.DataFrame) -> list[str]:
    """Datasets where Change frequency and Co-change agree on every sample."""
    out = []
    for dataset, g in srg.groupby("dataset"):
        pivot = g.pivot_table(
            index="sample_idx", columns="attribution_method", values="srg_score"
        )
        if {"Change frequency", "Co-change"} <= set(pivot.columns):
            delta = (pivot["Change frequency"] - pivot["Co-change"]).abs().max()
            if np.isfinite(delta) and delta == 0.0:
                out.append(str(dataset))
    return out


def write_report(notes: list[str], outputs: dict[str, pd.DataFrame], checks: pd.DataFrame) -> Path:
    lines = [
        "# Consolidation report",
        "",
        "Derived from `raw/` only. Every table below carries canonical method "
        "names and an explicit evaluation subset.",
        "",
        "## Outputs",
        "",
    ]
    for name, df in outputs.items():
        lines.append(f"- `{name}`: {len(df)} rows, {df.shape[1]} columns")
    lines += ["", "## Observations", ""]
    lines += notes or ["- none"]

    if not checks.empty:
        lines += ["", "## Checks against claims in the paper", ""]
        for _, row in checks.iterrows():
            mark = "ok" if row["agrees"] else "MISMATCH"
            lines.append(f"- [{mark}] {row['claim']} ({row['where']})")
            lines.append(f"  - paper: {row['paper_says']}")
            lines.append(f"  - data: {row['data_says']}")
    lines.append("")
    return write_text("\n".join(lines), CONSOLIDATED / "consolidation_report.md")




def run(verbose: bool = False) -> dict:
    log = setup_logging(verbose)
    CONSOLIDATED.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    wide = build_wide(notes)
    tidy = build_tidy(wide)
    validity = build_validity(wide)
    summary = build_metric_summary(wide)
    distribution = build_distribution(validity, notes)
    cmmd = build_cmmd_per_sample(wide)
    srg = build_srg_per_sample(notes)
    srg_curves, srg_mean_curves, srg_curve_summary = build_srg_curves(notes)
    config = build_search_config(notes)
    scm_effects, scm_models = build_scm(notes)
    summary_xcheck = build_summary_table_crosscheck(summary, notes)
    methods = method_table()

    tables: dict[str, pd.DataFrame] = {
        "method_names.csv": methods,
        "per_sample.csv": tidy,
        "per_sample_wide.csv": wide,
        "validity.csv": validity,
        "metric_summary.csv": summary,
        "distribution_set_level.csv": distribution,
        "cmmd_per_sample.csv": cmmd,
        "srg_per_sample.csv": srg,
        "srg_curves_per_sample.csv": srg_curves,
        "srg_mean_curves.csv": srg_mean_curves,
        "srg_curves_summary.csv": srg_curve_summary,
        "search_config.csv": config,
        "scm_effects.csv": scm_effects,
        "scm_model_level.csv": scm_models,
        "summary_table_crosscheck.csv": summary_xcheck,
    }

    outputs: list[Path] = []
    for name, df in tables.items():
        if df is None or df.empty:
            log.warning("skipping empty table %s", name)
            continue
        outputs.append(write_csv(df, CONSOLIDATED / name))

    checks = paper_crosscheck(validity, summary, srg, srg_curves, srg_curve_summary)
    if not checks.empty:
        outputs.append(write_csv(checks, CONSOLIDATED / "paper_crosscheck.csv"))
    outputs.append(write_report(notes, {k: v for k, v in tables.items() if not v.empty}, checks))

    stage = record_stage(
        "consolidate",
        outputs,
        extra={
            "n_tables": len(outputs),
            "mismatched_claims": int((~checks["agrees"]).sum()) if not checks.empty else 0,
        },
    )
    log.info("consolidate complete: %d files", len(outputs))
    return stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    run(verbose=args.verbose)


if __name__ == "__main__":
    main()
