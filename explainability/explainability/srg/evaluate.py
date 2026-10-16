"""Symmetric Relevance Gain (SRG) evaluation via pixel corruption."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..baselines.classifier import predict_class_prob, predict_class_probs_batch
from ..config import DatasetSpec
from ..phenotype.diffs import importance_to_ranking, spatial_shape, ensure_spatial_importance
from .occlusion import corrupt_pixels, parse_occlusion_strategy


@dataclass
class SRGResult:
    srg_score: float
    mif_curve: np.ndarray
    lif_curve: np.ndarray
    k_fractions: np.ndarray
    n_pixels: int
    p_baseline: float
    occlusion_strategy: str
    prob_class: int


def _curve_k_values(n_pixels: int, curve_points: int) -> np.ndarray:
    if curve_points >= n_pixels:
        return np.arange(n_pixels + 1, dtype=np.int64)
    return np.unique(
        np.round(np.linspace(0, n_pixels, num=curve_points + 1)).astype(np.int64)
    )


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_srg(
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    importance: np.ndarray,
    prob_class: int,
    weights: Path,
    device: torch.device,
    *,
    curve_points: int = 100,
    occlusion_strategy: str = "mean_alibi",
    occlusion_noise_std: float = 0.15,
    rng_seed: int = 0,
    infer_batch_size: int = 32,
) -> SRGResult:
    """Compute SRG = normalized area between LIF and MIF probability curves."""
    strategy = parse_occlusion_strategy(occlusion_strategy).value
    shape = spatial_shape(img_alibi)
    importance = ensure_spatial_importance(img_alibi, importance)
    n_pixels = int(np.prod(shape))
    ranking = importance_to_ranking(importance)
    rng = np.random.default_rng(rng_seed)

    p0 = predict_class_prob(spec, img_alibi, prob_class, weights, device)
    k_vals = _curve_k_values(n_pixels, curve_points)

    mif = np.zeros(len(k_vals), dtype=np.float64)
    lif = np.zeros(len(k_vals), dtype=np.float64)

    pending_images: list[np.ndarray] = []
    pending_meta: list[tuple[int, str]] = []

    for i, k in enumerate(k_vals):
        if k == 0:
            mif[i] = p0
            lif[i] = p0
            continue
        top_k = ranking[:k]
        bot_k = ranking[-k:]
        img_mif = corrupt_pixels(
            spec,
            img_alibi,
            top_k,
            strategy,
            noise_std=occlusion_noise_std,
            rng=rng,
        )
        img_lif = corrupt_pixels(
            spec,
            img_alibi,
            bot_k,
            strategy,
            noise_std=occlusion_noise_std,
            rng=rng,
        )
        pending_images.append(img_mif)
        pending_meta.append((i, "mif"))
        pending_images.append(img_lif)
        pending_meta.append((i, "lif"))

    if pending_images:
        probs = predict_class_probs_batch(
            spec,
            pending_images,
            prob_class,
            weights,
            device,
            batch_size=infer_batch_size,
        )
        for (curve_i, curve_kind), prob in zip(pending_meta, probs):
            if curve_kind == "mif":
                mif[curve_i] = prob
            else:
                lif[curve_i] = prob

    k_frac = k_vals.astype(np.float64) / max(n_pixels, 1)
    srg = _trapz(lif - mif, k_frac)
    return SRGResult(
        srg_score=srg,
        mif_curve=mif.astype(np.float32),
        lif_curve=lif.astype(np.float32),
        k_fractions=k_frac.astype(np.float32),
        n_pixels=n_pixels,
        p_baseline=p0,
        occlusion_strategy=strategy,
        prob_class=int(prob_class),
    )
