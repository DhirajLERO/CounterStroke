"""Interactive matplotlib figures for the metrics explorer (mirrors plot_methods)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .config_ui import COLOR_INK, COLOR_LAVENDER, COLOR_MINT, COLOR_PEACH
from .metrics_catalog import METRICS_BY_ID, MetricDef
from .metrics_io import COMPARE_DATASET_KEYS, DatasetMetricsBundle, metric_mean_err
from .metrics_preprocess import direction_for_column, ordered_methods_present

ChartStyle = Literal["auto", "bar", "boxplot", "twin_bar", "heatmap"]
ErrorBars = Literal["sem", "std", "none"]

_CHART_FACE = "#faf8ff"
_FOOTER_SUCCESS = (
    "Success: graco_* = o1 < 0; other methods = pred_class_cf != y_true."
)

_DATASET_COLORS = {
    "mnist": "#7EB6D9",
    "cifar10": "#A9D08E",
    "imagenet100": "#F4B183",
}
_DATASET_LABELS = {
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "imagenet100": "ImageNet-100",
}


@dataclass(frozen=True)
class PlotOptions:
    chart_style: ChartStyle = "auto"
    error_bars: ErrorBars = "sem"


def recommended_plot_kind(metric: MetricDef) -> str:
    if metric.id == "success_class_changed_rate":
        return "success_bar"
    if metric.csv_source == "per_sample":
        if metric.direction == "context":
            return "prob_panel"
        return "per_sample"
    if metric.csv_source == "distribution":
        return "distribution_bar"
    if metric.csv_source in ("batch_fid", "cmmd_batch", "cmmd_per_sample"):
        return "twin_bar"
    return "distribution_bar"


def twin_columns_for_metric(metric: MetricDef) -> tuple[str, str] | None:
    if metric.id in (
        "frechet_inception_distance_vs_source_y_true",
        "frechet_inception_distance_vs_predicted_class",
    ):
        return (
            "frechet_inception_distance_vs_source_y_true",
            "frechet_inception_distance_vs_predicted_class",
        )
    if metric.id in ("cmmd_vs_stratified_source_y_true", "cmmd_vs_stratified_predicted_class"):
        return ("cmmd_vs_stratified_source_y_true", "cmmd_vs_stratified_predicted_class")
    if metric.id in ("cmmd_vs_source_y_true_class", "cmmd_vs_predicted_class"):
        return ("cmmd_vs_source_y_true_class", "cmmd_vs_predicted_class")
    return None


def _style_ax(ax) -> None:
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors=COLOR_INK)


def _boxplot(ax, data: list, methods: list[str]) -> None:
    try:
        ax.boxplot(data, tick_labels=methods, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=methods, showmeans=True)


def _error_values(ser: pd.Series, error_bars: ErrorBars) -> float:
    if error_bars == "none":
        return 0.0
    if error_bars == "std":
        return float(ser.std(ddof=1)) if ser.notna().sum() > 1 else 0.0
    return float(ser.sem()) if ser.notna().sum() > 1 else 0.0


def _filter_methods_df(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    if not methods or "method" not in df.columns:
        return df
    return df[df["method"].isin(methods)]


def figure_to_pdf_bytes(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.4)
    buf.seek(0)
    return buf.read()


def fig_compare_datasets(
    metric_ids: list[str],
    bundles_by_ds: dict[str, DatasetMetricsBundle],
    methods: list[str],
    *,
    error_bars: ErrorBars = "sem",
) -> Figure | None:
    """Multi-panel grouped bars: x=methods, series=datasets (one panel per metric)."""
    metrics = [METRICS_BY_ID[m] for m in metric_ids if m in METRICS_BY_ID]
    if not metrics or not methods:
        return None

    ds_keys = [k for k in COMPARE_DATASET_KEYS if k in bundles_by_ds]
    if not ds_keys:
        return None

    n = len(metrics)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(max(10.0, 0.55 * len(methods) + 4.0), max(3.2, 2.8 * n)),
        sharex=True,
        squeeze=False,
        facecolor=_CHART_FACE,
    )

    n_ds = len(ds_keys)
    width = min(0.8 / max(n_ds, 1), 0.28)
    x = np.arange(len(methods))

    for i, metric in enumerate(metrics):
        ax = axes[i][0]
        dir_hint = {
            "lower": "lower is better",
            "higher": "higher is better",
            "context": "contextual",
        }.get(metric.direction, metric.direction)
        scale_success = metric.id == "success_class_changed_rate"

        for j, ds in enumerate(ds_keys):
            means_errs = metric_mean_err(
                bundles_by_ds[ds],
                metric,
                methods,
                error_bars=error_bars,
            )
            by_m = {
                str(r["method"]): (float(r["mean"]), float(r["err"]))
                for _, r in means_errs.iterrows()
            }
            heights = []
            errs = []
            for m in methods:
                if m not in by_m:
                    heights.append(np.nan)
                    errs.append(0.0)
                    continue
                mean, err = by_m[m]
                if scale_success:
                    mean *= 100.0
                    err *= 100.0
                heights.append(mean)
                errs.append(err if error_bars != "none" else 0.0)

            offset = (j - (n_ds - 1) / 2.0) * width
            ax.bar(
                x + offset,
                heights,
                width=width,
                yerr=errs if error_bars != "none" else None,
                capsize=2,
                label=_DATASET_LABELS.get(ds, ds) if i == 0 else None,
                color=_DATASET_COLORS.get(ds, COLOR_LAVENDER),
                edgecolor="#ddd8ea",
                linewidth=0.6,
            )

        ylabel = "Success rate (%)" if scale_success else metric.label
        ax.set_ylabel(ylabel, color=COLOR_INK, fontsize=9)
        ax.set_title(f"{metric.label} ({dir_hint})", color=COLOR_INK, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        _style_ax(ax)

    axes[-1][0].set_xticks(x)
    axes[-1][0].set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(ds_keys),
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle("Cross-dataset comparison (from tables)", fontsize=12, y=1.06, color=COLOR_INK)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
    return fig


def fig_success_rate(df_valid: pd.DataFrame, methods: list[str]) -> Figure:
    sub = _filter_methods_df(df_valid, methods)
    if "success_plot" in sub.columns:
        succ = sub.groupby("method", sort=False)["success_plot"].mean()
    else:
        succ = (
            sub.assign(_chg=sub["pred_class_cf"] != sub["y_true"])
            .groupby("method", sort=False)["_chg"]
            .mean()
        )
    ordered = [m for m in ordered_methods_present(succ) if m in methods] if methods else ordered_methods_present(succ)
    ordered = [m for m in ordered if m in succ.index]
    vals = [float(succ[m]) * 100 for m in ordered]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_CHART_FACE)
    x = np.arange(len(ordered))
    colors = [COLOR_LAVENDER, COLOR_MINT, COLOR_PEACH] * (len(ordered) // 3 + 1)
    bars = ax.bar(x, vals, color=colors[: len(ordered)], edgecolor="#ddd8ea")
    ax.set_ylabel("Success rate (%)", color=COLOR_INK)
    ax.set_title("Class changed on counterfactual", color=COLOR_INK, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, max(105, max(vals, default=0) * 1.1))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    fig.text(0.5, 0.01, _FOOTER_SUCCESS, ha="center", fontsize=7, color=COLOR_INK)
    return fig


def fig_single_column_bar(
    df: pd.DataFrame,
    col: str,
    methods: list[str],
    *,
    title: str,
    direction: str,
    error_bars: ErrorBars,
) -> Figure:
    sub = _filter_methods_df(df, methods)
    ordered = [m for m in ordered_methods_present(sub) if not methods or m in methods]
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=_CHART_FACE)
    x = np.arange(len(ordered))
    heights, errs = [], []
    for m in ordered:
        ser = sub.loc[sub["method"] == m, col].dropna()
        heights.append(float(ser.mean()) if len(ser) else np.nan)
        errs.append(_error_values(ser, error_bars))
    colors = [COLOR_LAVENDER, COLOR_MINT, COLOR_PEACH] * (len(ordered) // 3 + 1)
    ax.bar(
        x,
        heights,
        yerr=errs if error_bars != "none" else None,
        capsize=3,
        color=colors[: len(ordered)],
        edgecolor="#ddd8ea",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(col, color=COLOR_INK)
    ax.set_title(f"{title} ({direction} is better)", color=COLOR_INK, fontsize=10)
    _style_ax(ax)
    fig.tight_layout()
    return fig


def fig_single_column_boxplot(
    df: pd.DataFrame,
    col: str,
    methods: list[str],
    *,
    title: str,
    direction: str,
) -> Figure:
    sub = _filter_methods_df(df, methods)
    ordered = [m for m in ordered_methods_present(sub) if not methods or m in methods]
    data = [sub.loc[sub["method"] == m, col].dropna().values for m in ordered]
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=_CHART_FACE)
    _boxplot(ax, data, ordered)
    ax.set_title(f"{title} ({direction} is better)", color=COLOR_INK, fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    _style_ax(ax)
    fig.tight_layout()
    return fig


def fig_distribution_bar(
    dm: pd.DataFrame,
    col: str,
    methods: list[str],
    *,
    title: str,
) -> Figure | None:
    if col not in dm.columns:
        return None
    ordered = [m for m in ordered_methods_present(dm) if not methods or m in methods]
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=_CHART_FACE)
    x = np.arange(len(ordered))
    vals = [
        float(dm.loc[dm["method"] == m, col].iloc[0]) if (dm["method"] == m).any() else np.nan
        for m in ordered
    ]
    colors = [COLOR_LAVENDER, COLOR_MINT, COLOR_PEACH] * (len(ordered) // 3 + 1)
    ax.bar(x, vals, color=colors[: len(ordered)], edgecolor="#ddd8ea")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(col, color=COLOR_INK)
    ax.set_title(title, color=COLOR_INK, fontsize=10)
    _style_ax(ax)
    fig.tight_layout()
    return fig


def fig_distribution_heatmap(dm: pd.DataFrame, methods: list[str]) -> Figure | None:
    num_cols = [c for c in dm.select_dtypes(include=[np.number]).columns if c != "n_paired"]
    if not num_cols:
        return None
    sub = _filter_methods_df(dm, methods)
    ordered = [m for m in ordered_methods_present(sub) if not methods or m in methods]
    mat = np.zeros((len(ordered), len(num_cols)))
    for i, m in enumerate(ordered):
        row = sub.loc[sub["method"] == m, num_cols]
        if row.empty:
            mat[i, :] = np.nan
        else:
            mat[i, :] = row.iloc[0].values.astype(float)
    z = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        mu = np.nanmean(col)
        sig = np.nanstd(col)
        if sig and not np.isnan(sig):
            z[:, j] = -(col - mu) / sig
    fig, ax = plt.subplots(
        figsize=(max(10, len(num_cols) * 1.5), max(5, len(ordered) * 0.7)),
        facecolor=_CHART_FACE,
    )
    im = ax.imshow(z, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=9)
    ax.set_title("Distribution metrics (z-scored; brighter = better)", color=COLOR_INK, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    return fig


def fig_twin_bar(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    methods: list[str],
    *,
    title: str,
    label_a: str,
    label_b: str,
) -> Figure | None:
    if col_a not in df.columns or col_b not in df.columns:
        return None
    ordered = [m for m in ordered_methods_present(df) if not methods or m in methods]
    x = np.arange(len(ordered))
    w = 0.36
    v1 = [float(df.loc[df["method"] == m, col_a].iloc[0]) if (df["method"] == m).any() else np.nan for m in ordered]
    v2 = [float(df.loc[df["method"] == m, col_b].iloc[0]) if (df["method"] == m).any() else np.nan for m in ordered]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_CHART_FACE)
    ax.bar(x - w / 2, v1, width=w, label=label_a, color=COLOR_LAVENDER, edgecolor="#ddd8ea")
    ax.bar(x + w / 2, v2, width=w, label=label_b, color=COLOR_MINT, edgecolor="#ddd8ea")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_title(title, color=COLOR_INK, fontsize=10)
    ax.legend(fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    return fig


def fig_cmmd_per_sample_twin(df: pd.DataFrame, methods: list[str]) -> Figure | None:
    c1, c2 = "cmmd_vs_source_y_true_class", "cmmd_vs_predicted_class"
    if c1 not in df.columns or c2 not in df.columns:
        return None
    sub = df.loc[df["has_valid_cf"] == 1] if "has_valid_cf" in df.columns else df
    sub = _filter_methods_df(sub, methods)
    ordered = [m for m in ordered_methods_present(sub) if not methods or m in methods]
    g = sub.groupby("method")[[c1, c2]].mean()
    x = np.arange(len(ordered))
    w = 0.36
    v1 = [float(g.loc[m, c1]) if m in g.index else np.nan for m in ordered]
    v2 = [float(g.loc[m, c2]) if m in g.index else np.nan for m in ordered]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_CHART_FACE)
    ax.bar(x - w / 2, v1, width=w, label=c1, color=COLOR_PEACH, edgecolor="#ddd8ea")
    ax.bar(x + w / 2, v2, width=w, label=c2, color=COLOR_MINT, edgecolor="#ddd8ea")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_title("Per-sample CMMD means (valid CF rows)", color=COLOR_INK, fontsize=10)
    ax.legend(fontsize=7)
    _style_ax(ax)
    fig.tight_layout()
    return fig


def _resolve_style(metric: MetricDef, opts: PlotOptions) -> str:
    if opts.chart_style == "heatmap":
        return "heatmap"
    if opts.chart_style == "boxplot":
        return "boxplot"
    if opts.chart_style == "twin_bar":
        return "twin"
    if opts.chart_style == "bar":
        return "bar"
    kind = recommended_plot_kind(metric)
    if kind in ("success_bar", "prob_panel", "per_sample", "distribution_bar"):
        return "bar"
    if kind == "twin_bar":
        return "twin"
    return "bar"


def build_figure(
    metric: MetricDef,
    bundle: DatasetMetricsBundle,
    methods: list[str],
    *,
    opts: PlotOptions | None = None,
) -> Figure | None:
    opts = opts or PlotOptions()
    style = _resolve_style(metric, opts)
    df_valid = bundle.df_valid
    frames = bundle.frames

    if metric.id == "success_class_changed_rate":
        if df_valid is None or df_valid.empty:
            return None
        return fig_success_rate(df_valid, methods)

    if style == "heatmap" and metric.csv_source == "distribution" and frames.get("distribution") is not None:
        return fig_distribution_heatmap(frames["distribution"], methods)

    use_twin = style == "twin" or (
        opts.chart_style == "auto" and recommended_plot_kind(metric) == "twin_bar"
    )
    if use_twin:
        twins = twin_columns_for_metric(metric)
        if metric.csv_source == "batch_fid" and frames.get("batch_fid") is not None and twins:
            return fig_twin_bar(
                frames["batch_fid"],
                twins[0],
                twins[1],
                methods,
                title="Batch FID: stratified real pools vs CF set",
                label_a="vs source y_true",
                label_b="vs predicted class",
            )
        if metric.csv_source == "cmmd_batch" and frames.get("cmmd_batch") is not None and twins:
            return fig_twin_bar(
                frames["cmmd_batch"],
                twins[0],
                twins[1],
                methods,
                title="Stratified batch CMMD",
                label_a="vs stratified source",
                label_b="vs stratified predicted",
            )
        if metric.csv_source == "cmmd_per_sample" and frames.get("cmmd_per_sample") is not None:
            return fig_cmmd_per_sample_twin(frames["cmmd_per_sample"], methods)

    if metric.csv_source == "batch_fid" and frames.get("batch_fid") is not None and metric.csv_column:
        return fig_distribution_bar(frames["batch_fid"], metric.csv_column, methods, title=metric.label)

    if metric.csv_source == "cmmd_batch" and frames.get("cmmd_batch") is not None and metric.csv_column:
        return fig_distribution_bar(frames["cmmd_batch"], metric.csv_column, methods, title=metric.label)

    if metric.csv_source == "cmmd_per_sample" and frames.get("cmmd_per_sample") is not None and metric.csv_column:
        cm = frames["cmmd_per_sample"]
        if "has_valid_cf" in cm.columns:
            cm = cm.loc[cm["has_valid_cf"] == 1]
        col = metric.csv_column
        if col in cm.columns:
            if style == "boxplot":
                return fig_single_column_boxplot(cm, col, methods, title=metric.label, direction="lower")
            sub = _filter_methods_df(cm, methods)
            ordered = [m for m in ordered_methods_present(sub) if not methods or m in methods]
            g = sub.groupby("method", sort=False)[col].mean()
            fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=_CHART_FACE)
            x = np.arange(len(ordered))
            vals = [float(g[m]) if m in g.index else np.nan for m in ordered]
            colors = [COLOR_LAVENDER, COLOR_MINT, COLOR_PEACH] * (len(ordered) // 3 + 1)
            ax.bar(x, vals, color=colors[: len(ordered)], edgecolor="#ddd8ea")
            ax.set_xticks(x)
            ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
            ax.set_title(metric.label, color=COLOR_INK, fontsize=10)
            _style_ax(ax)
            fig.tight_layout()
            return fig

    if metric.csv_source == "distribution" and frames.get("distribution") is not None:
        col = metric.csv_column
        if col:
            return fig_distribution_bar(frames["distribution"], col, methods, title=metric.label)

    if df_valid is None or metric.csv_column is None:
        return None
    col = metric.csv_column
    if col not in df_valid.columns:
        return None

    direction = metric.direction if metric.direction != "context" else direction_for_column(col)
    if style == "boxplot":
        return fig_single_column_boxplot(df_valid, col, methods, title=metric.label, direction=direction)
    return fig_single_column_bar(
        df_valid,
        col,
        methods,
        title=metric.label,
        direction=direction,
        error_bars=opts.error_bars,
    )
