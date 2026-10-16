"""Factual final-population filtering, distributions, and change maps."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .datasets import AppDatasetSpec
from .population_io import IndividualRecord
from .render import record_to_alibi

PROB_THRESHOLD = 0.50
EPS = 1e-6


class ChangeMapMethod(str, Enum):
    SUM_INTENSITY = "sum_intensity"
    CHANGE_FREQUENCY = "change_frequency"
    MEAN_INTENSITY = "mean_intensity"
    MAX_INTENSITY = "max_intensity"
    COCHANGE_STRICT = "cochange_strict"
    SPATIAL_COHERENCE = "spatial_coherence"


def filter_flipped_high_conf(
    records: list[IndividualRecord],
    *,
    prob_threshold: float = PROB_THRESHOLD,
) -> list[IndividualRecord]:
    return [
        r
        for r in records
        if r.pred_class != r.original_class and r.pred_prob >= prob_threshold
    ]


def pred_class_distribution(records: list[IndividualRecord]) -> Counter[int]:
    return Counter(r.pred_class for r in records)


@dataclass
class FactualAnalysisResult:
    n_total: int
    n_filtered: int
    top_class: int | None
    top_count: int
    distribution: dict[int, int]
    filtered: list[IndividualRecord]


def analyse_factual_distribution(
    records: list[IndividualRecord],
    *,
    prob_threshold: float = PROB_THRESHOLD,
) -> FactualAnalysisResult:
    filtered = filter_flipped_high_conf(records, prob_threshold=prob_threshold)
    dist = pred_class_distribution(filtered)
    if not dist:
        return FactualAnalysisResult(
            n_total=len(records),
            n_filtered=0,
            top_class=None,
            top_count=0,
            distribution={},
            filtered=[],
        )
    top_class, top_count = dist.most_common(1)[0]
    return FactualAnalysisResult(
        n_total=len(records),
        n_filtered=len(filtered),
        top_class=top_class,
        top_count=top_count,
        distribution=dict(dist),
        filtered=filtered,
    )


def _diff_scalar(orig: np.ndarray, cf: np.ndarray) -> np.ndarray:
    diff = np.abs(cf - orig)
    if diff.ndim == 3 and diff.shape[0] == 3:
        diff = np.transpose(diff, (1, 2, 0))
    if diff.ndim == 3:
        return diff.max(axis=-1)
    return diff


def _spatial_shape(orig: np.ndarray) -> tuple[int, ...]:
    if orig.ndim == 3 and orig.shape[0] == 3:
        return (orig.shape[1], orig.shape[2])
    if orig.ndim == 3:
        return (orig.shape[0], orig.shape[1])
    return orig.shape


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    m = float(arr.max())
    if m > 0:
        return (arr / m).astype(np.float32)
    return arr.astype(np.float32)


def _blur_2d(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(arr, sigma=sigma).astype(np.float32)
    except ImportError:
        # lightweight box blur fallback
        k = 5
        pad = k // 2
        padded = np.pad(arr, pad, mode="edge")
        out = np.zeros_like(arr, dtype=np.float32)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                out[i, j] = padded[i : i + k, j : j + k].mean()
        return out


def _collect_diffs(
    spec: AppDatasetSpec,
    orig_alibi: np.ndarray,
    records: list[IndividualRecord],
    *,
    target_pred_class: int,
    max_render: int,
) -> tuple[list[np.ndarray], int]:
    subset = [r for r in records if r.pred_class == target_pred_class]
    if not subset:
        return [], 0
    if len(subset) > max_render:
        subset = subset[:max_render]
    orig = np.asarray(orig_alibi, dtype=np.float32)
    diffs: list[np.ndarray] = []
    for rec in subset:
        try:
            cf = record_to_alibi(spec, rec, orig)
        except Exception:
            continue
        d = _diff_scalar(orig, cf)
        if np.any(d > EPS):
            diffs.append(d)
    return diffs, len(diffs)


def heat_from_diffs(
    diffs: list[np.ndarray],
    shape: tuple[int, ...],
    method: str | ChangeMapMethod,
    *,
    cochange_tau: float = 0.5,
) -> np.ndarray:
    """Build normalized heatmap from precomputed per-CF diff scalars."""
    if not diffs:
        return np.zeros(shape, dtype=np.float32)

    m = ChangeMapMethod(method) if isinstance(method, str) else method
    n = len(diffs)

    if m == ChangeMapMethod.CHANGE_FREQUENCY:
        count = np.zeros(shape, dtype=np.float64)
        for d in diffs:
            count[d > EPS] += 1.0
        return _normalize_01(count / max(n, 1))

    if m == ChangeMapMethod.MAX_INTENSITY:
        accum = np.zeros(shape, dtype=np.float64)
        for d in diffs:
            accum = np.maximum(accum, d)
        return _normalize_01(accum)

    if m == ChangeMapMethod.MEAN_INTENSITY:
        sum_d = np.zeros(shape, dtype=np.float64)
        cnt = np.zeros(shape, dtype=np.float64)
        for d in diffs:
            mask = d > EPS
            sum_d[mask] += d[mask]
            cnt[mask] += 1.0
        with np.errstate(invalid="ignore"):
            mean = np.where(cnt > 0, sum_d / np.maximum(cnt, 1), 0)
        return _normalize_01(mean)

    if m == ChangeMapMethod.COCHANGE_STRICT:
        freq = np.zeros(shape, dtype=np.float64)
        for d in diffs:
            freq[d > EPS] += 1.0
        frac = freq / max(n, 1)
        out = np.where(frac >= cochange_tau, frac, frac * 0.5)
        return _normalize_01(out)

    if m == ChangeMapMethod.SPATIAL_COHERENCE:
        freq = np.zeros(shape, dtype=np.float64)
        for d in diffs:
            freq[d > EPS] += 1.0
        frac = (freq / max(n, 1)).astype(np.float32)
        blurred = _blur_2d(frac, sigma=1.5)
        return _normalize_01(blurred)

    accum = np.zeros(shape, dtype=np.float64)
    for d in diffs:
        mask = d > EPS
        accum[mask] += d[mask]
    return _normalize_01(accum)


def build_change_map_method(
    spec: AppDatasetSpec,
    orig_alibi: np.ndarray,
    records: list[IndividualRecord],
    *,
    target_pred_class: int,
    method: str | ChangeMapMethod = ChangeMapMethod.SUM_INTENSITY,
    max_render: int = 200,
    cochange_tau: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Normalized heatmap in [0, 1] and count of CFs used."""
    shape = _spatial_shape(orig_alibi)
    diffs, n_used = _collect_diffs(
        spec, orig_alibi, records, target_pred_class=target_pred_class, max_render=max_render
    )
    if not diffs:
        return np.zeros(shape, dtype=np.float32), 0
    heat = heat_from_diffs(diffs, shape, method, cochange_tau=cochange_tau)
    return heat, n_used


def collect_diffs_for_target(
    spec: AppDatasetSpec,
    orig_alibi: np.ndarray,
    records: list[IndividualRecord],
    *,
    target_pred_class: int,
    max_render: int = 200,
) -> tuple[list[np.ndarray], int, tuple[int, ...]]:
    """Return diffs once for building several heatmap methods."""
    shape = _spatial_shape(orig_alibi)
    diffs, n_used = _collect_diffs(
        spec, orig_alibi, records, target_pred_class=target_pred_class, max_render=max_render
    )
    return diffs, n_used, shape


def build_change_map(
    spec: AppDatasetSpec,
    orig_alibi: np.ndarray,
    records: list[IndividualRecord],
    *,
    target_pred_class: int,
    max_render: int = 200,
) -> tuple[np.ndarray, int]:
    return build_change_map_method(
        spec,
        orig_alibi,
        records,
        target_pred_class=target_pred_class,
        method=ChangeMapMethod.SUM_INTENSITY,
        max_render=max_render,
    )
