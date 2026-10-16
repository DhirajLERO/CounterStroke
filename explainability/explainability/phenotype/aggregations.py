"""Change-map aggregations ported from analyse_final_population."""

from __future__ import annotations

from enum import Enum

import numpy as np

from .diffs import EPS, importance_to_ranking


class PhenotypeMapMethod(str, Enum):
    SUM_INTENSITY = "sum_intensity"
    MEAN_INTENSITY = "mean_intensity"
    CHANGE_FREQUENCY = "change_frequency"
    COCHANGE_STRICT = "cochange_strict"
    SPATIAL_COHERENCE = "spatial_coherence"


def normalize_01(arr: np.ndarray) -> np.ndarray:
    m = float(arr.max())
    if m > 0:
        return (arr / m).astype(np.float32)
    return arr.astype(np.float32)


def blur_2d(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(arr, sigma=sigma).astype(np.float32)
    except ImportError:
        k = 5
        pad = k // 2
        padded = np.pad(arr, pad, mode="edge")
        out = np.zeros_like(arr, dtype=np.float32)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                out[i, j] = padded[i : i + k, j : j + k].mean()
        return out


def heat_from_diffs(
    diffs: list[np.ndarray],
    shape: tuple[int, int],
    method: str | PhenotypeMapMethod,
    *,
    cochange_tau: float = 0.5,
    spatial_sigma: float = 1.5,
    eps: float = EPS,
) -> np.ndarray:
    if not diffs:
        return np.zeros(shape, dtype=np.float32)

    m = PhenotypeMapMethod(method) if isinstance(method, str) else method
    n = len(diffs)

    if m == PhenotypeMapMethod.CHANGE_FREQUENCY:
        count = np.zeros(shape, dtype=np.float64)
        for d in diffs:
            count[d > eps] += 1.0
        return normalize_01(count / max(n, 1))

    if m == PhenotypeMapMethod.MEAN_INTENSITY:
        sum_d = np.zeros(shape, dtype=np.float64)
        cnt = np.zeros(shape, dtype=np.float64)
        for d in diffs:
            mask = d > eps
            sum_d[mask] += d[mask]
            cnt[mask] += 1.0
        with np.errstate(invalid="ignore"):
            mean = np.where(cnt > 0, sum_d / np.maximum(cnt, 1), 0)
        return normalize_01(mean)

    if m == PhenotypeMapMethod.COCHANGE_STRICT:
        freq = np.zeros(shape, dtype=np.float64)
        for d in diffs:
            freq[d > eps] += 1.0
        frac = freq / max(n, 1)
        out = np.where(frac >= cochange_tau, frac, frac * 0.5)
        return normalize_01(out)

    if m == PhenotypeMapMethod.SPATIAL_COHERENCE:
        freq = np.zeros(shape, dtype=np.float64)
        for d in diffs:
            freq[d > eps] += 1.0
        frac = (freq / max(n, 1)).astype(np.float32)
        blurred = blur_2d(frac, sigma=spatial_sigma)
        return normalize_01(blurred)

    accum = np.zeros(shape, dtype=np.float64)
    for d in diffs:
        mask = d > eps
        accum[mask] += d[mask]
    return normalize_01(accum)


PHENOTYPE_METHODS = [m.value for m in PhenotypeMapMethod]
