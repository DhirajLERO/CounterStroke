"""Prepare per-sample metrics DataFrames (mirror plot_methods preprocessing)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics_catalog import METHOD_ORDER

GRACO_METHODS = frozenset({"graco_base_o2", "graco_base_o3", "graco50"})

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


def direction_for_column(name: str) -> str:
    if name in METRIC_DIRECTION:
        return METRIC_DIRECTION[name]
    if name.startswith("lpips_") or name.startswith("inception_sqdist_"):
        return "lower"
    return "context"


def ordered_methods_present(series_or_df: pd.Series | pd.DataFrame, col: str = "method") -> list[str]:
    if isinstance(series_or_df, pd.Series):
        present = set(series_or_df.index.dropna().unique())
    else:
        present = set(series_or_df[col].dropna().unique())
    out = [m for m in METHOD_ORDER if m in present]
    out.extend(sorted(present - set(out)))
    return out


def _load_graco_strategy_valid_map(gen_root: Path) -> dict[int, dict[str, int]]:
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
            try:
                row[str(m)] = int(gsv.get(m, 0))
            except (TypeError, ValueError):
                row[str(m)] = 0
        out[sid] = row
    return out


def _load_graco_selected_row_map(gen_root: Path) -> dict[int, dict[str, object]]:
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
    try:
        return float(ent.get("o1")) < 0.0
    except (TypeError, ValueError):
        return False


def _baseline_success_class_changed(pred_cf: object, y_true: object) -> bool:
    if pd.isna(pred_cf):
        return False
    try:
        return int(pred_cf) != int(y_true)
    except (ValueError, TypeError):
        return False


def apply_success_plot_column(df: pd.DataFrame, gen_root: Path) -> pd.DataFrame:
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


def apply_graco_has_valid_cf_from_meta(df: pd.DataFrame, gen_root: Path) -> pd.DataFrame:
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


def per_sample_numeric_columns(df: pd.DataFrame) -> list[str]:
    skip = {
        "sample_idx",
        "y_true",
        "y_intended_target",
        "method",
        "has_valid_cf",
        "pred_class_cf",
        "success_plot",
    }
    cols: list[str] = []
    for c in df.columns:
        if c in skip or pd.api.types.is_bool_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def classify_per_sample_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    num = per_sample_numeric_columns(df)
    lower_cols = [c for c in num if direction_for_column(c) == "lower"]
    higher_cols = [c for c in num if direction_for_column(c) == "higher"]
    prob_cols = [c for c in num if direction_for_column(c) == "context"]
    return lower_cols, higher_cols, prob_cols


def prepare_per_sample_df(df: pd.DataFrame, gen_root: Path) -> pd.DataFrame:
    """Apply Graco meta overrides, success_plot, return valid-CF rows only."""
    df = apply_graco_has_valid_cf_from_meta(df, gen_root)
    df = apply_success_plot_column(df, gen_root)
    return df.loc[df["has_valid_cf"] == 1].copy()
