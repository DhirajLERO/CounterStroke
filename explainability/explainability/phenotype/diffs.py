"""Per-CF diff collection from phenotype populations."""

from __future__ import annotations

import numpy as np

from ..config import DatasetSpec
from ..data.population import IndividualRecord
from ..render.turtle import record_to_alibi

EPS = 1e-6


def diff_scalar(orig: np.ndarray, cf: np.ndarray) -> np.ndarray:
    diff = np.abs(cf - orig)
    if diff.ndim == 3 and diff.shape[0] == 3:
        diff = np.transpose(diff, (1, 2, 0))
    if diff.ndim == 3:
        return diff.max(axis=-1)
    return diff


def spatial_shape(orig: np.ndarray) -> tuple[int, int]:
    if orig.ndim == 3 and orig.shape[0] == 3:
        return (orig.shape[1], orig.shape[2])
    if orig.ndim == 3:
        return (orig.shape[0], orig.shape[1])
    h, w = orig.shape
    return (h, w)


def attribution_to_spatial_map(img_alibi: np.ndarray, attribution: np.ndarray) -> np.ndarray:
    """Reduce attribution tensors to H×W matching *img_alibi* (handles CHW/HWC/NCHW/NHWC)."""
    hw = spatial_shape(img_alibi)
    h, w = hw
    arr = np.asarray(attribution, dtype=np.float32)

    while arr.ndim > len(hw) and arr.shape[0] == 1:
        arr = arr[0]

    if arr.shape == hw:
        return np.abs(arr)

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[1:] == hw:
            return np.abs(arr).max(axis=0)
        if arr.shape[:2] == hw and arr.shape[2] in (1, 3):
            return np.abs(arr).max(axis=-1)

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            return attribution_to_spatial_map(img_alibi, arr[0])
        if arr.shape[1] in (1, 3) and arr.shape[2:] == hw:
            return np.abs(arr[0]).max(axis=0)
        if arr.shape[1:3] == hw and arr.shape[3] in (1, 3):
            return np.abs(arr[0]).max(axis=-1)

    if arr.size == h * w:
        return np.abs(arr).reshape(hw)

    raise ValueError(
        f"Cannot map attribution shape {arr.shape} to spatial {hw} "
        f"({h * w} pixels; got {arr.size} values)"
    )


def ensure_spatial_importance(img_alibi: np.ndarray, importance: np.ndarray) -> np.ndarray:
    """Ensure an importance map is H×W before SRG ranking/occlusion."""
    hw = spatial_shape(img_alibi)
    imp = np.asarray(importance, dtype=np.float32)
    if imp.shape == hw:
        return imp
    if imp.size == hw[0] * hw[1]:
        return imp.reshape(hw)
    return attribution_to_spatial_map(img_alibi, imp)


def collect_diffs(
    spec: DatasetSpec,
    orig_alibi: np.ndarray,
    records: list[IndividualRecord],
    *,
    target_pred_class: int,
    max_render: int = 200,
    eps: float = EPS,
) -> tuple[list[np.ndarray], int, tuple[int, int]]:
    subset = [r for r in records if r.pred_class == target_pred_class]
    if not subset:
        shape = spatial_shape(orig_alibi)
        return [], 0, shape
    if len(subset) > max_render:
        subset = subset[:max_render]
    orig = np.asarray(orig_alibi, dtype=np.float32)
    diffs: list[np.ndarray] = []
    for rec in subset:
        try:
            cf = record_to_alibi(spec, rec, orig)
        except Exception:
            continue
        d = diff_scalar(orig, cf)
        if np.any(d > eps):
            diffs.append(d.astype(np.float32))
    return diffs, len(diffs), spatial_shape(orig)


def importance_to_ranking(importance: np.ndarray) -> np.ndarray:
    flat = importance.ravel()
    return np.argsort(-flat, kind="stable")
