"""SSIM between counterfactual and reference images in [0, 1]."""

from __future__ import annotations

import numpy as np
import torch


def _to_nchw_01(img: np.ndarray) -> torch.Tensor:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
    elif a.ndim == 3 and a.shape[-1] in (1, 3):
        chw = np.transpose(a, (2, 0, 1))
        t = torch.from_numpy(chw).unsqueeze(0)
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
    else:
        raise ValueError(f"Expected HWC or HW, got {a.shape}")
    return t


def best_ssim_match(
    cf_01: np.ndarray,
    pool_01: list[np.ndarray],
    *,
    device: torch.device | None = None,
) -> tuple[int, float]:
    """Return (index in pool, SSIM score) for the best-matching reference."""
    if not pool_01:
        return -1, float("nan")

    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

    dev = device or torch.device("cpu")
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    cf_t = _to_nchw_01(cf_01).to(dev)

    best_i = 0
    best_s = -1.0
    with torch.no_grad():
        for i, ref in enumerate(pool_01):
            ref_t = _to_nchw_01(ref).to(dev)
            s = float(metric(cf_t, ref_t).item())
            if s > best_s:
                best_s = s
                best_i = i
    return best_i, best_s
