"""Load precomputed metric CSVs and locate pipeline plot assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from .datasets import AppDatasetSpec
from .metrics_catalog import MetricDef
from .metrics_paths import MetricsDirs, resolve_metrics_dirs
from .metrics_preprocess import prepare_per_sample_df


@dataclass(frozen=True)
class DatasetMetricsBundle:
    spec: AppDatasetSpec
    dirs: MetricsDirs
    frames: dict[str, pd.DataFrame | None]
    df_valid: pd.DataFrame | None


@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str, mtime: float) -> pd.DataFrame | None:
    _ = mtime
    p = Path(path_str)
    if not p.is_file():
        return None
    return pd.read_csv(p)


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    mtime = path.stat().st_mtime
    return load_csv_cached(str(path.resolve()), mtime)


def load_all_metrics_csvs(dirs: MetricsDirs) -> dict[str, pd.DataFrame | None]:
    return {
        "per_sample": _load_csv(dirs.gen_root / "per_sample_metrics.csv"),
        "distribution": _load_csv(dirs.gen_root / "distribution_metrics.csv"),
        "batch_fid": _load_csv(dirs.gen_root / "batch_fid.csv"),
        "cmmd_per_sample": _load_csv(dirs.gen_root / "cmmd_per_sample.csv"),
        "cmmd_batch": _load_csv(dirs.gen_root / "cmmd_batch_stratified.csv"),
        "summary": _load_csv(dirs.results_dir / "summary_tables.csv"),
    }


def load_dataset_bundle(
    spec: AppDatasetSpec,
    *,
    gen_override: str | Path | None = None,
    results_override: str | Path | None = None,
) -> DatasetMetricsBundle:
    dirs = resolve_metrics_dirs(
        spec,
        gen_override=gen_override,
        results_override=results_override,
    )
    frames = load_all_metrics_csvs(dirs)
    df_valid = None
    ps = frames.get("per_sample")
    if ps is not None and not ps.empty:
        df_valid = prepare_per_sample_df(ps, dirs.gen_root)
    return DatasetMetricsBundle(spec=spec, dirs=dirs, frames=frames, df_valid=df_valid)


COMPARE_DATASET_KEYS = ("mnist", "cifar10", "imagenet100")


def load_all_dataset_bundles(
    overrides: dict[str, dict[str, str]] | None = None,
) -> dict[str, DatasetMetricsBundle]:
    """Load metric bundles for MNIST / CIFAR-10 / ImageNet-100."""
    from .datasets import get_app_dataset_spec

    overrides = overrides or {}
    out: dict[str, DatasetMetricsBundle] = {}
    for name in COMPARE_DATASET_KEYS:
        ov = overrides.get(name, {})
        gen = ov.get("gen") or None
        results = ov.get("results") or None
        if gen is not None and not str(gen).strip():
            gen = None
        if results is not None and not str(results).strip():
            results = None
        out[name] = load_dataset_bundle(
            get_app_dataset_spec(name),
            gen_override=gen,
            results_override=results,
        )
    return out


def methods_union(bundles: dict[str, DatasetMetricsBundle]) -> list[str]:
    """Ordered union of methods across bundles."""
    from .metrics_catalog import METHOD_ORDER

    present: set[str] = set()
    for bundle in bundles.values():
        present.update(methods_in_data(bundle.df_valid))
        present.update(methods_in_data(bundle.frames.get("distribution")))
        present.update(methods_in_data(bundle.frames.get("batch_fid")))
        present.update(methods_in_data(bundle.frames.get("cmmd_batch")))
        present.update(methods_in_data(bundle.frames.get("cmmd_per_sample")))
    out = [m for m in METHOD_ORDER if m in present]
    out.extend(sorted(present - set(out)))
    return out


def summary_table_for_metric(
    metric: MetricDef,
    bundle: DatasetMetricsBundle,
    methods: list[str],
) -> pd.DataFrame:
    """Same aggregation used by the metrics UI tables."""
    if metric.csv_source == "per_sample" and bundle.df_valid is not None:
        return aggregate_by_method(bundle.df_valid, metric, methods=methods)
    if metric.csv_source == "distribution" and metric.csv_column:
        return distribution_table(bundle.frames.get("distribution"), metric.csv_column, methods)
    if metric.csv_source == "batch_fid" and metric.csv_column:
        return distribution_table(bundle.frames.get("batch_fid"), metric.csv_column, methods)
    if metric.csv_source == "cmmd_batch" and metric.csv_column:
        return distribution_table(bundle.frames.get("cmmd_batch"), metric.csv_column, methods)
    if metric.csv_source == "cmmd_per_sample" and metric.csv_column:
        df = bundle.frames.get("cmmd_per_sample")
        if df is not None and "has_valid_cf" in df.columns:
            df = df.loc[df["has_valid_cf"] == 1]
        if df is not None and metric.csv_column in df.columns:
            sub = df[df["method"].isin(methods)] if methods else df
            g = sub.groupby("method", sort=False)[metric.csv_column].agg(["mean", "std", "count"])
            return g.reset_index()
    summary = summary_rows_for_metric(metric, bundle.frames.get("summary"), methods)
    if not summary.empty:
        return summary
    return pd.DataFrame()


def metric_mean_err(
    bundle: DatasetMetricsBundle,
    metric: MetricDef,
    methods: list[str],
    *,
    error_bars: str = "sem",
) -> pd.DataFrame:
    """Return DataFrame with columns method, mean, err for one metric."""
    table = summary_table_for_metric(metric, bundle, methods)
    if table.empty or "method" not in table.columns:
        return pd.DataFrame(columns=["method", "mean", "err"])

    rows: list[dict] = []
    for method in methods:
        sub = table.loc[table["method"] == method]
        if sub.empty:
            continue
        row = sub.iloc[0]
        if "mean" in table.columns:
            mean = float(row["mean"])
            std = float(row["std"]) if "std" in table.columns and pd.notna(row.get("std")) else 0.0
            count = float(row["count"]) if "count" in table.columns and pd.notna(row.get("count")) else 0.0
            if error_bars == "none":
                err = 0.0
            elif error_bars == "std":
                err = std if count > 1 else 0.0
            else:
                err = (std / (count**0.5)) if count > 1 else 0.0
        else:
            col = metric.csv_column
            if col and col in table.columns and pd.notna(row.get(col)):
                mean = float(row[col])
            elif "value" in table.columns and pd.notna(row.get("value")):
                mean = float(row["value"])
            else:
                continue
            err = 0.0
        rows.append({"method": method, "mean": mean, "err": err})

    return pd.DataFrame(rows)


def methods_in_data(df: pd.DataFrame | None) -> list[str]:
    if df is None or "method" not in df.columns:
        return []
    from .metrics_catalog import METHOD_ORDER

    present = set(df["method"].dropna().unique())
    out = [m for m in METHOD_ORDER if m in present]
    out.extend(sorted(present - set(out)))
    return out


def filter_methods(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    if not methods or "method" not in df.columns:
        return df
    return df[df["method"].isin(methods)]


def plot_path_for_stem(results_dir: Path, stem: str, ext: str = "jpg") -> Path | None:
    for suffix in (ext, "jpg", "png"):
        p = results_dir / f"{stem}.{suffix}"
        if p.is_file():
            return p
    return None


def pdf_path_for_stem(results_dir: Path, stem: str) -> Path | None:
    p = results_dir / f"{stem}.pdf"
    return p if p.is_file() else None


def aggregate_by_method(
    df: pd.DataFrame,
    metric: MetricDef,
    *,
    methods: list[str],
) -> pd.DataFrame:
    sub = filter_methods(df, methods)
    if metric.id == "success_class_changed_rate":
        if "success_plot" in sub.columns:
            g = sub.groupby("method", sort=False)["success_plot"].agg(["mean", "std", "count"])
        elif "pred_class_cf" in sub.columns and "y_true" in sub.columns:
            sub = sub.assign(_ok=sub["pred_class_cf"] != sub["y_true"])
            g = sub.groupby("method", sort=False)["_ok"].agg(["mean", "std", "count"])
        else:
            return pd.DataFrame()
        return g.reset_index()

    col = metric.csv_column
    if not col or col not in sub.columns:
        return pd.DataFrame()
    return sub.groupby("method", sort=False)[col].agg(["mean", "std", "count"]).reset_index()


def summary_rows_for_metric(
    metric: MetricDef,
    summary_df: pd.DataFrame | None,
    methods: list[str],
) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
    key = metric.csv_column or metric.id
    if "metric" not in summary_df.columns:
        return pd.DataFrame()
    rows = summary_df[summary_df["metric"] == key]
    if metric.id == "success_class_changed_rate":
        rows = summary_df[summary_df["metric"] == "success_class_changed_rate"]
    if methods and "method" in rows.columns:
        rows = rows[rows["method"].isin(methods)]
    return rows


def distribution_table(
    df: pd.DataFrame | None,
    col: str,
    methods: list[str],
) -> pd.DataFrame:
    if df is None or col not in df.columns:
        return pd.DataFrame()
    sub = filter_methods(df, methods)
    cols = ["method", col]
    if "n_paired" in sub.columns:
        cols.append("n_paired")
    return sub[cols]
