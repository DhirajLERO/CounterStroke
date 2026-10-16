"""SRG pixel corruption / occlusion strategies (Alibi space)."""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..config import DatasetSpec
from ..phenotype.diffs import spatial_shape

# White in display [0, 1] → Alibi 0.5; black → -0.5
ALIBI_WHITE = 0.5
ALIBI_BLACK = -0.5


class OcclusionStrategy(str, Enum):
    MEAN_ALIBI = "mean_alibi"
    INVERT_ALIBI = "invert_alibi"
    WHITE = "white"
    BLACK = "black"
    GAUSSIAN_NOISE = "gaussian_noise"


OCCLUSION_STRATEGIES = tuple(s.value for s in OcclusionStrategy)


def parse_occlusion_strategy(name: str) -> OcclusionStrategy:
    try:
        return OcclusionStrategy(name)
    except ValueError as exc:
        valid = ", ".join(OCCLUSION_STRATEGIES)
        raise ValueError(f"Unknown occlusion strategy {name!r}; choose one of: {valid}") from exc


def mean_alibi_value(spec: DatasetSpec, img_alibi: np.ndarray) -> float | np.ndarray:
    arr = np.asarray(img_alibi, dtype=np.float32)
    if spec.name == "imagenet100" and arr.ndim == 3 and arr.shape[0] == 3:
        return np.mean(arr, axis=(1, 2)).astype(np.float32)
    return float(np.mean(arr))


def _set_pixel(
    out: np.ndarray,
    spec: DatasetSpec,
    r: int,
    c: int,
    value: float | np.ndarray,
) -> None:
    if spec.name == "imagenet100" and out.ndim == 3 and out.shape[0] == 3:
        if isinstance(value, np.ndarray):
            out[:, r, c] = value
        else:
            out[:, r, c] = float(value)
    elif out.ndim == 3:
        out[r, c, ...] = value
    else:
        out[r, c] = float(value)


def _white_value(spec: DatasetSpec) -> float | np.ndarray:
    if spec.name == "imagenet100":
        return np.full(3, ALIBI_WHITE, dtype=np.float32)
    return ALIBI_WHITE


def _black_value(spec: DatasetSpec) -> float | np.ndarray:
    if spec.name == "imagenet100":
        return np.full(3, ALIBI_BLACK, dtype=np.float32)
    return ALIBI_BLACK


def _noise_value(
    spec: DatasetSpec,
    rng: np.random.Generator,
    noise_std: float,
) -> float | np.ndarray:
    if spec.name == "imagenet100":
        v = rng.normal(0.0, noise_std, size=3).astype(np.float32)
        return np.clip(v, ALIBI_BLACK, ALIBI_WHITE)
    return float(np.clip(rng.normal(0.0, noise_std), ALIBI_BLACK, ALIBI_WHITE))


def corrupt_pixels(
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    flat_spatial_indices: np.ndarray,
    strategy: str | OcclusionStrategy,
    *,
    noise_std: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a copy of ``img_alibi`` with selected spatial pixels occluded."""
    if len(flat_spatial_indices) == 0:
        return np.array(img_alibi, dtype=np.float32, copy=True)

    strat = parse_occlusion_strategy(strategy) if isinstance(strategy, str) else strategy
    out = np.array(img_alibi, dtype=np.float32, copy=True)
    h, w = spatial_shape(img_alibi)
    rng = rng or np.random.default_rng(0)

    mean_fill = mean_alibi_value(spec, img_alibi)
    white_fill = _white_value(spec)
    black_fill = _black_value(spec)

    for idx in flat_spatial_indices:
        r, c = divmod(int(idx), w)
        if strat == OcclusionStrategy.MEAN_ALIBI:
            _set_pixel(out, spec, r, c, mean_fill)
        elif strat == OcclusionStrategy.INVERT_ALIBI:
            if spec.name == "imagenet100" and out.ndim == 3 and out.shape[0] == 3:
                _set_pixel(out, spec, r, c, -out[:, r, c])
            elif out.ndim == 3:
                out[r, c, ...] = -out[r, c, ...]
            else:
                out[r, c] = -out[r, c]
        elif strat == OcclusionStrategy.WHITE:
            _set_pixel(out, spec, r, c, white_fill)
        elif strat == OcclusionStrategy.BLACK:
            _set_pixel(out, spec, r, c, black_fill)
        elif strat == OcclusionStrategy.GAUSSIAN_NOISE:
            _set_pixel(out, spec, r, c, _noise_value(spec, rng, noise_std))
        else:
            raise ValueError(strat)

    return out
