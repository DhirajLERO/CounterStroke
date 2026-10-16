"""Compare counterfactual methods and write plots."""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHOD_ORDER = [
    "counterfactual_pt",
    "cfproto",
    "cfrl",
    "leapfactual",
    "leapfactual_r",
    "graco_base_o2",
    "graco_base_o3",
    "graco50",
]

GRACO_METHODS = frozenset({"graco_base_o2", "graco_base_o3", "graco50"})

# Appended to footers that reference has_valid_cf (Graco uses meta.json, not CSV tensor validity).
FOOTER_GRACO_HAS_VALID_CF = (
    "Graco (graco_*): has_valid_cf follows sample_*/meta.json → graco_strategy_valid "
    "(strategy had a qualifying phenotype in pooled Graco CSVs), not evaluator tensor checks in the CSV."
)

FOOTER_SUCCESS_DEFINITION = (
    "Success: graco_* = selected-row o1 < 0 (Graco CSV in meta.json); "
    "other methods = pred_class_cf != y_true (evaluator on saved tensor)."
)

# Per-sample column -> "lower" | "higher" | "context"
METRIC_DIRECTION: dict[str, str] = {
    "l2_distance": "lower",
    "sparsity_count": "lower",
    "sparsity_frac": "lower",
    "ssim_to_original": "higher",
    "ssim_to_random_pred_class": "higher",
    "p_true_orig": "context",
    "p_true_cf": "context",
    "delta_p_true": "context",
    "p_intended_target_cf": "context",
    "p_pred_cf": "context",
}


def _direction_for_column(name: str) -> str:
    if name in METRIC_DIRECTION:
        return METRIC_DIRECTION[name]
    if name.startswith("lpips_") or name.startswith("inception_sqdist_"):
        return "lower"
    return "context"


def _ordered_methods_present(series_or_df: pd.Series | pd.DataFrame, col: str = "method") -> list[str]:
    if isinstance(series_or_df, pd.Series):
        present = set(series_or_df.index.dropna().unique())
    else:
        present = set(series_or_df[col].dropna().unique())
    out = [m for m in METHOD_ORDER if m in present]
    out.extend(sorted(present - set(out)))
    return out


def _direction_caption(direction: str) -> str:
    if direction == "lower":
        return "Lower is better"
    if direction == "higher":
        return "Higher is better"
    return "Interpretation depends on goal"


def _direction_badge_text(direction: str) -> str:
    if direction == "lower":
        return "↓ Lower better"
    if direction == "higher":
        return "↑ Higher better"
    return "Context-dependent"


def _wrap_footer_lines(lines: list[str], width: int = 96) -> list[str]:
    out: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        if len(line) <= width:
            out.append(line)
        else:
            out.extend(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False))
    return out


def _load_graco_strategy_valid_map(gen_root: Path) -> dict[int, dict[str, int]]:
    """sample_idx -> { method: 0|1 } from export_gracoi_to_genroot meta.json."""
    out: dict[int, dict[str, int]] = {}
    for meta_path in sorted(gen_root.glob("sample_*/meta.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        gsv = data.get("graco_strategy_valid")
        if not isinstance(gsv, dict):
            continue
        sid = data.get("sample_idx")
        if sid is None:
            try:
                sid = int(meta_path.parent.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
        sid = int(sid)
        row: dict[str, int] = {}
        for m in GRACO_METHODS:
            v = gsv.get(m, 0)
            try:
                row[str(m)] = int(v)
            except (TypeError, ValueError):
                row[str(m)] = 0
        out[sid] = row
    return out


def _load_graco_selected_row_map(gen_root: Path) -> dict[int, dict[str, object]]:
    """sample_idx -> graco_selected_row dict from export meta.json."""
    out: dict[int, dict[str, object]] = {}
    for meta_path in sorted(gen_root.glob("sample_*/meta.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        gsr = data.get("graco_selected_row")
        if not isinstance(gsr, dict):
            continue
        sid = data.get("sample_idx")
        if sid is None:
            try:
                sid = int(meta_path.parent.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
        out[int(sid)] = gsr
    return out


def _graco_success_o1(gsr: dict[str, object], method: str) -> bool:
    ent = gsr.get(method)
    if not isinstance(ent, dict):
        return False
    o1 = ent.get("o1")
    try:
        return float(o1) < 0.0
    except (TypeError, ValueError):
        return False


def _baseline_success_class_changed(pred_cf: object, y_true: object) -> bool:
    if pd.isna(pred_cf):
        return False
    try:
        return int(pred_cf) != int(y_true)
    except (ValueError, TypeError):
        return False


def _apply_success_plot_column(df: pd.DataFrame, gen_root: Path) -> pd.DataFrame:
    """success_plot: graco_* uses meta graco_selected_row o1 < 0; else evaluator class change."""
    smap = _load_graco_selected_row_map(gen_root)
    df = df.copy()
    vals: list[bool] = []
    for _, r in df.iterrows():
        m = str(r["method"])
        if m in GRACO_METHODS:
            vals.append(_graco_success_o1(smap.get(int(r["sample_idx"]), {}), m))
        else:
            vals.append(_baseline_success_class_changed(r["pred_class_cf"], r["y_true"]))
    df["success_plot"] = vals
    return df


def _apply_graco_has_valid_cf_from_meta(df: pd.DataFrame, gen_root: Path) -> pd.DataFrame:
    """Override has_valid_cf for graco_* rows using meta.json (plan: strategy-qualified phenotype)."""
    gsv = _load_graco_strategy_valid_map(gen_root)
    if not gsv:
        return df
    mask = df["method"].isin(GRACO_METHODS)
    if not mask.any():
        return df
    df = df.copy()
    new_vals = [
        int(gsv.get(int(sid), {}).get(str(m), 0))
        for sid, m in zip(df.loc[mask, "sample_idx"], df.loc[mask, "method"])
    ]
    df.loc[mask, "has_valid_cf"] = new_vals
    return df


def _fig_footer(
    fig,
    lines: list[str],
    *,
    multiax: bool = False,
    top: float | None = None,
    right: float | None = None,
) -> None:
    """Reserve bottom margin, draw wrapped footer lines; use wider spacing for multi-axis figures."""
    wrapped = _wrap_footer_lines(lines)
    n = max(1, len(wrapped))
    line_frac = 0.028
    bottom = min(0.38, 0.09 + n * line_frac)

    if multiax:
        if top is None:
            top = 0.90
        fig.subplots_adjust(
            left=0.07,
            right=0.98,
            top=top,
            bottom=bottom,
            hspace=0.56,
            wspace=0.42,
        )
    else:
        if top is None:
            top = 0.94
        fig.subplots_adjust(
            left=0.10,
            right=0.96 if right is None else right,
            top=top,
            bottom=bottom,
        )

    y = 0.012
    for line in reversed(wrapped):
        fig.text(0.5, y, line, ha="center", va="bottom", fontsize=8, color="0.15")
        y += line_frac


def _save_figure(fig, out_path: Path) -> None:
    """Save every plot in both JPEG and PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")
    fig.savefig(stem.with_suffix(".jpg"), dpi=150, bbox_inches="tight", pad_inches=0.5)
    fig.savefig(stem.with_suffix(".pdf"), dpi=150, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)


def _boxplot(ax, data: list, methods: list[str]) -> None:
    try:
        ax.boxplot(data, tick_labels=methods, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=methods, showmeans=True)


def plot_success_rate(df_valid: pd.DataFrame, out_path: Path) -> None:
    chg_col = "success_plot" if "success_plot" in df_valid.columns else None
    if chg_col is None:
        succ = (
            df_valid.assign(_chg=df_valid["pred_class_cf"] != df_valid["y_true"])
            .groupby("method", sort=False)["_chg"]
            .mean()
        )
    else:
        succ = df_valid.groupby("method", sort=False)[chg_col].mean()
    methods = _ordered_methods_present(succ)
    vals = [float(succ[m]) * 100 for m in methods]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.arange(len(methods))
    bars = ax.bar(x, vals, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Class changed on counterfactual", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, max(105, max(vals, default=0) * 1.1))
    ax.margins(x=0.02)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    _fig_footer(
        fig,
        [
            _direction_caption("higher")
            + " — success definition varies by method (see next lines).",
            "Among rows with has_valid_cf = 1 only.",
            FOOTER_GRACO_HAS_VALID_CF,
            FOOTER_SUCCESS_DEFINITION,
        ],
    )
    _save_figure(fig, out_path)


def _per_sample_numeric_columns(df: pd.DataFrame) -> list[str]:
    skip = {
        "sample_idx",
        "y_true",
        "y_intended_target",
        "method",
        "has_valid_cf",
        "pred_class_cf",
        "success_plot",  # bool; matplotlib boxplot needs float quantiles
    }
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def plot_metric_grid(
    df_valid: pd.DataFrame,
    columns: list[str],
    out_path: Path,
    kind: str,
    *,
    boxplot: bool,
) -> None:
    if not columns:
        return
    methods = _ordered_methods_present(df_valid)
    n = len(columns)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig_w = max(14.0, 4.6 * ncols)
    fig_h = max(10.0, 4.0 * nrows + 1.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    direction = kind if kind in ("lower", "higher") else "context"
    for i, col in enumerate(columns):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if boxplot:
            data = [df_valid.loc[df_valid["method"] == m, col].dropna().values for m in methods]
            _boxplot(ax, data, methods)
            ax.set_title(col, fontsize=10, pad=8)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
        else:
            means = df_valid.groupby("method")[col].mean()
            sem = df_valid.groupby("method")[col].sem()
            x = np.arange(len(methods))
            h = [means.get(m, np.nan) for m in methods]
            err = [sem.get(m, np.nan) for m in methods]
            ax.bar(x, h, yerr=err, capsize=3, color="cadetblue", edgecolor="black", linewidth=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
            ax.set_title(col, fontsize=10, pad=8)
        subdir = _direction_for_column(col) if kind == "context" else direction
        ax.text(
            0.98,
            0.02,
            _direction_badge_text(subdir),
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="wheat", alpha=0.85, edgecolor="0.5", linewidth=0.4),
        )
    for j in range(len(columns), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    prefix = "Mean ± SEM" if not boxplot else "Distribution"
    fig.suptitle(
        f"{prefix}: per-sample metrics ({_direction_caption(direction)})",
        fontsize=12,
        y=0.98,
    )
    _fig_footer(
        fig,
        [
            f"Valid CF rows only (has_valid_cf = 1). Panel: {_direction_caption(direction)}.",
            FOOTER_GRACO_HAS_VALID_CF,
        ],
        multiax=True,
        top=0.93,
    )
    _save_figure(fig, out_path)


def plot_probabilities_panel(df_valid: pd.DataFrame, out_path: Path, boxplot: bool) -> None:
    prob_cols = [c for c in _per_sample_numeric_columns(df_valid) if _direction_for_column(c) == "context"]
    if not prob_cols:
        return
    methods = _ordered_methods_present(df_valid)
    n = len(prob_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig_w = max(13.0, 4.8 * ncols)
    fig_h = max(9.0, 4.0 * nrows + 1.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    for i, col in enumerate(prob_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if boxplot:
            data = [df_valid.loc[df_valid["method"] == m, col].dropna().values for m in methods]
            _boxplot(ax, data, methods)
        else:
            means = df_valid.groupby("method")[col].mean()
            sem = df_valid.groupby("method")[col].sem()
            x = np.arange(len(methods))
            h = [means.get(m, np.nan) for m in methods]
            err = [sem.get(m, np.nan) for m in methods]
            ax.bar(x, h, yerr=err, capsize=3, color="darkseagreen", edgecolor="black", linewidth=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(col, fontsize=10, pad=8)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.text(
            0.98,
            0.02,
            _direction_badge_text("context"),
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow", alpha=0.9, edgecolor="0.5", linewidth=0.4),
        )
    for j in range(len(prob_cols), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    kind = "boxplots" if boxplot else "means"
    fig.suptitle(f"Classifier probabilities ({kind})", fontsize=12, y=0.98)
    _fig_footer(
        fig,
        [
            "Context-dependent: after a class change you often want lower p_true_cf; "
            "higher p_intended_target_cf if that matches your experimental target.",
            "See analysis/README.md for column meanings.",
            FOOTER_GRACO_HAS_VALID_CF,
        ],
        multiax=True,
        top=0.93,
    )
    _save_figure(fig, out_path)


def plot_distribution_bar(dm: pd.DataFrame, col: str, out_path: Path, title: str, footer_extra: str) -> bool:
    if col not in dm.columns:
        return False
    methods = _ordered_methods_present(dm)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(methods))
    vals = [float(dm.loc[dm["method"] == m, col].iloc[0]) if (dm["method"] == m).any() else np.nan for m in methods]
    ax.bar(x, vals, color="slateblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(col, fontsize=9)
    ax.set_title(title, fontsize=12, pad=10)
    ax.margins(x=0.02)
    _fig_footer(fig, [_direction_caption("lower") + ". " + footer_extra])
    _save_figure(fig, out_path)
    return True


def plot_distribution_heatmap(dm: pd.DataFrame, out_path: Path) -> None:
    num_cols = [c for c in dm.select_dtypes(include=[np.number]).columns if c != "n_paired"]
    if not num_cols:
        return
    methods = _ordered_methods_present(dm)
    mat = np.zeros((len(methods), len(num_cols)))
    for i, m in enumerate(methods):
        row = dm.loc[dm["method"] == m, num_cols]
        if row.empty:
            mat[i, :] = np.nan
        else:
            mat[i, :] = row.iloc[0].values.astype(float)
    # z-score per column; all distribution metrics here are lower-better -> negate so higher = better
    z = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        mu = np.nanmean(col)
        sig = np.nanstd(col)
        if sig and not np.isnan(sig):
            z[:, j] = -(col - mu) / sig
        else:
            z[:, j] = 0.0
    fig_w = max(12.0, len(num_cols) * 2.0)
    fig_h = max(7.0, len(methods) * 0.85 + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(z, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_title(
        "Distribution metrics (z-scored; sign flipped so brighter = better on this scale)",
        fontsize=11,
        pad=14,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Higher = better after normalization")
    cbar.ax.tick_params(labelsize=8)
    _fig_footer(
        fig,
        [
            "Each column z-scored across methods; lower-better raw metrics are negated so larger values are preferable here.",
            "See analysis/README.md for whether each raw metric is lower- or higher-better.",
        ],
        top=0.90,
        right=0.86,
    )
    _save_figure(fig, out_path)


def plot_batch_fid(bf: pd.DataFrame, out_path: Path) -> None:
    c1 = "frechet_inception_distance_vs_source_y_true"
    c2 = "frechet_inception_distance_vs_predicted_class"
    if c1 not in bf.columns or c2 not in bf.columns:
        return False
    methods = _ordered_methods_present(bf)
    x = np.arange(len(methods))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 6.0))
    v1 = [float(bf.loc[bf["method"] == m, c1].iloc[0]) if (bf["method"] == m).any() else np.nan for m in methods]
    v2 = [float(bf.loc[bf["method"] == m, c2].iloc[0]) if (bf["method"] == m).any() else np.nan for m in methods]
    ax.bar(x - w / 2, v1, width=w, label="vs source y_true pool", color="coral", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, v2, width=w, label="vs predicted-class pool", color="skyblue", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Fréchet Inception Distance")
    ax.set_title("Batch FID: stratified real pools vs all valid CFs per method", fontsize=12, pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ax.margins(x=0.02)
    _fig_footer(
        fig,
        [
            _direction_caption("lower") + " — CF set closer to the referenced real-digit pools is better.",
            "Data: batch_fid.csv.",
        ],
    )
    _save_figure(fig, out_path)
    return True


def plot_cmmd_batch_stratified(cb: pd.DataFrame, out_path: Path) -> None:
    c1 = "cmmd_vs_stratified_source_y_true"
    c2 = "cmmd_vs_stratified_predicted_class"
    if c1 not in cb.columns or c2 not in cb.columns:
        return False
    methods = _ordered_methods_present(cb)
    x = np.arange(len(methods))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 6.0))
    v1 = [float(cb.loc[cb["method"] == m, c1].iloc[0]) if (cb["method"] == m).any() else np.nan for m in methods]
    v2 = [float(cb.loc[cb["method"] == m, c2].iloc[0]) if (cb["method"] == m).any() else np.nan for m in methods]
    ax.bar(x - w / 2, v1, width=w, label="vs stratified source y_true", color="peru", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, v2, width=w, label="vs stratified predicted class", color="olivedrab", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("CLIP MMD (CMMD-style)")
    ax.set_title("Stratified train reals vs CF set", fontsize=12, pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ax.margins(x=0.02)
    _fig_footer(fig, [_direction_caption("lower") + ". Data: cmmd_batch_stratified.csv."])
    _save_figure(fig, out_path)
    return True


def plot_cmmd_per_sample_means(cm: pd.DataFrame, out_path: Path) -> None:
    c1, c2 = "cmmd_vs_source_y_true_class", "cmmd_vs_predicted_class"
    if c1 not in cm.columns or c2 not in cm.columns:
        return False
    sub = cm.loc[cm["has_valid_cf"] == 1].copy()
    methods = _ordered_methods_present(sub)
    g = sub.groupby("method")[[c1, c2]].mean()
    x = np.arange(len(methods))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 6.0))
    v1 = [float(g.loc[m, c1]) if m in g.index else np.nan for m in methods]
    v2 = [float(g.loc[m, c2]) if m in g.index else np.nan for m in methods]
    ax.bar(x - w / 2, v1, width=w, label=c1, color="tan", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, v2, width=w, label=c2, color="darkkhaki", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean per-sample CMMD")
    ax.set_title("Per-sample CMMD averaged by method (valid CF rows)", fontsize=12, pad=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.margins(x=0.02)
    _fig_footer(fig, [_direction_caption("lower") + ". Data: cmmd_per_sample.csv."])
    _save_figure(fig, out_path)
    return True


def build_summary_tables(
    df_valid: pd.DataFrame,
    optional_frames: dict[str, pd.DataFrame | None],
    out_csv: Path,
) -> None:
    rows: list[dict[str, object]] = []
    if "success_plot" in df_valid.columns:
        succ = df_valid.groupby("method", sort=False)["success_plot"].mean()
    else:
        succ = (
            df_valid.assign(_chg=df_valid["pred_class_cf"] != df_valid["y_true"])
            .groupby("method", sort=False)["_chg"]
            .mean()
        )
    for m in _ordered_methods_present(succ):
        if m not in succ.index:
            continue
        rows.append(
            {
                "method": m,
                "metric": "success_class_changed_rate",
                "aggregate": "mean",
                "value": float(succ[m]),
                "n": int((df_valid["method"] == m).sum()),
                "direction": "higher",
            }
        )
    num_cols = _per_sample_numeric_columns(df_valid)
    for col in num_cols:
        d = _direction_for_column(col)
        g = df_valid.groupby("method")[col]
        for m in _ordered_methods_present(df_valid):
            if m not in g.groups:
                continue
            ser = g.get_group(m)
            rows.append(
                {
                    "method": m,
                    "metric": col,
                    "aggregate": "mean",
                    "value": float(ser.mean()),
                    "n": int(ser.notna().sum()),
                    "direction": d,
                }
            )
            rows.append(
                {
                    "method": m,
                    "metric": col,
                    "aggregate": "std",
                    "value": float(ser.std(ddof=1)) if ser.notna().sum() > 1 else float("nan"),
                    "n": int(ser.notna().sum()),
                    "direction": d,
                }
            )
    dm = optional_frames.get("distribution_metrics")
    if dm is not None:
        for col in dm.select_dtypes(include=[np.number]).columns:
            if col == "n_paired":
                continue
            for _, r in dm.iterrows():
                rows.append(
                    {
                        "method": r["method"],
                        "metric": col,
                        "aggregate": "value",
                        "value": float(r[col]),
                        "n": int(r.get("n_paired", np.nan)) if pd.notna(r.get("n_paired", np.nan)) else "",
                        "direction": "lower",
                    }
                )
    bf = optional_frames.get("batch_fid")
    if bf is not None:
        for _, r in bf.iterrows():
            for col in bf.select_dtypes(include=[np.number]).columns:
                rows.append(
                    {
                        "method": r["method"],
                        "metric": f"batch_fid.{col}",
                        "aggregate": "value",
                        "value": float(r[col]),
                        "n": "",
                        "direction": "lower",
                    }
                )
    cb = optional_frames.get("cmmd_batch_stratified")
    if cb is not None:
        for _, r in cb.iterrows():
            for col in cb.select_dtypes(include=[np.number]).columns:
                if col == "n_paired":
                    continue
                rows.append(
                    {
                        "method": r["method"],
                        "metric": f"cmmd_batch_stratified.{col}",
                        "aggregate": "value",
                        "value": float(r[col]),
                        "n": int(r.get("n_paired", np.nan)) if pd.notna(r.get("n_paired", np.nan)) else "",
                        "direction": "lower",
                    }
                )
    cm = optional_frames.get("cmmd_per_sample")
    if cm is not None:
        sub = cm.loc[cm["has_valid_cf"] == 1]
        for col in ["cmmd_vs_source_y_true_class", "cmmd_vs_predicted_class"]:
            if col not in sub.columns:
                continue
            g = sub.groupby("method")[col].mean()
            for m in _ordered_methods_present(sub):
                if m not in g.index:
                    continue
                rows.append(
                    {
                        "method": m,
                        "metric": f"cmmd_per_sample_mean.{col}",
                        "aggregate": "mean",
                        "value": float(g[m]),
                        "n": int((sub["method"] == m).sum()),
                        "direction": "lower",
                    }
                )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CF methods from metric CSVs.")
    parser.add_argument(
        "--gen-root",
        type=Path,
        default=Path("outputs/cf100_train_graco"),
        help="Directory with per_sample_metrics.csv etc. (e.g. merged baseline + Graco tensors).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Plot output directory. Default: <parent of gen-root>/results/<basename(gen-root)> "
        "(e.g. .../Cifar10/outputs/results/generation when gen-root is .../outputs/generation).",
    )
    args = parser.parse_args()
    gen_root = args.gen_root.resolve()
    if not gen_root.is_dir():
        print(f"gen-root is not a directory: {gen_root}", file=sys.stderr)
        return 1
    out_dir = args.out_dir
    if out_dir is None:
        # Same layout for every dataset: e.g. Cifar10/outputs/generation → Cifar10/outputs/results/generation
        out_dir = gen_root.parent / "results" / gen_root.name
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ps_path = gen_root / "per_sample_metrics.csv"
    if not ps_path.exists():
        print(f"Required file missing: {ps_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(ps_path)
    if "has_valid_cf" not in df.columns or "pred_class_cf" not in df.columns or "y_true" not in df.columns:
        print("per_sample_metrics.csv missing required columns.", file=sys.stderr)
        return 1
    df = _apply_graco_has_valid_cf_from_meta(df, gen_root)
    df = _apply_success_plot_column(df, gen_root)
    df_valid = df.loc[df["has_valid_cf"] == 1].copy()

    optional: dict[str, pd.DataFrame | None] = {
        "distribution_metrics": None,
        "batch_fid": None,
        "cmmd_batch_stratified": None,
        "cmmd_per_sample": None,
    }
    for key, name in [
        ("distribution_metrics", "distribution_metrics.csv"),
        ("batch_fid", "batch_fid.csv"),
        ("cmmd_batch_stratified", "cmmd_batch_stratified.csv"),
        ("cmmd_per_sample", "cmmd_per_sample.csv"),
    ]:
        p = gen_root / name
        if p.exists():
            optional[key] = pd.read_csv(p)
        else:
            print(f"Optional file missing (skipping related plots): {p}", file=sys.stderr)

    plot_success_rate(df_valid, out_dir / "01_success_rate_class_changed.png")

    num_cols = _per_sample_numeric_columns(df_valid)
    lower_cols = [c for c in num_cols if _direction_for_column(c) == "lower"]
    higher_cols = [c for c in num_cols if _direction_for_column(c) == "higher"]

    plot_metric_grid(df_valid, lower_cols, out_dir / "02_per_sample_means_grid_lower.png", "lower", boxplot=False)
    plot_metric_grid(df_valid, higher_cols, out_dir / "02_per_sample_means_grid_higher.png", "higher", boxplot=False)
    plot_probabilities_panel(df_valid, out_dir / "04_probabilities_panel_means.png", boxplot=False)

    plot_metric_grid(df_valid, lower_cols, out_dir / "03_per_sample_boxplots_lower.png", "lower", boxplot=True)
    plot_metric_grid(df_valid, higher_cols, out_dir / "03_per_sample_boxplots_higher.png", "higher", boxplot=True)
    plot_probabilities_panel(df_valid, out_dir / "03_per_sample_boxplots_probabilities.png", boxplot=True)

    dm = optional["distribution_metrics"]
    if dm is not None:
        specs = [
            (
                "frechet_inception_distance_cf_vs_originals",
                out_dir / "10_distribution_frechet_paired_orig_cf.png",
                "Paired originals vs CFs: Fréchet Inception Distance",
                "Smaller = CF distribution closer to paired originals in Inception space.",
            ),
            (
                "kid_mean_cf_vs_originals",
                out_dir / "11_distribution_kid_mean.png",
                "KID mean (CF vs originals)",
                "Kernel Inception Distance vs training originals; lower = closer.",
            ),
            (
                "kid_std_cf_vs_originals",
                out_dir / "12_distribution_kid_std.png",
                "KID std (bootstrap)",
                "Lower usually means less estimator variance across KID subsets.",
            ),
            (
                "cmmd_orig_vs_cf_distribution",
                out_dir / "13_distribution_cmmd_paired.png",
                "CLIP MMD (paired orig vs CF distribution)",
                "CMMD-style distance between original and CF sets in CLIP space.",
            ),
        ]
        for col, path, title, foot in specs:
            if not plot_distribution_bar(dm, col, path, title, foot):
                print(f"Column {col!r} not in distribution_metrics; skipped {path.name}", file=sys.stderr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            plot_distribution_heatmap(dm, out_dir / "14_distribution_metrics_heatmap.png")

    build_summary_tables(df_valid, optional, out_dir / "summary_tables.csv")

    bf = optional["batch_fid"]
    if bf is not None:
        if not plot_batch_fid(bf, out_dir / "20_batch_fid_stratified.png"):
            print("batch_fid.csv: expected columns missing.", file=sys.stderr)

    cb = optional["cmmd_batch_stratified"]
    if cb is not None:
        if not plot_cmmd_batch_stratified(cb, out_dir / "21_cmmd_batch_stratified.png"):
            print("cmmd_batch_stratified.csv: expected columns missing.", file=sys.stderr)

    cm = optional["cmmd_per_sample"]
    if cm is not None:
        if not plot_cmmd_per_sample_means(cm, out_dir / "22_cmmd_per_sample_means.png"):
            print("cmmd_per_sample.csv: expected columns missing.", file=sys.stderr)

    print(f"Wrote plots and summary_tables.csv under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
