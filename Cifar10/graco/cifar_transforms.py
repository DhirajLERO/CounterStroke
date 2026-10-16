"""CIFAR-10 RGB tensors in Alibi [-0.5, 0.5] (HWC)."""

from __future__ import annotations

import numpy as np


def alibi_to_01_rgb(img_hwc: np.ndarray) -> np.ndarray:
    """(H,W,3) Alibi → [0,1] for visualization."""
    return np.clip(np.asarray(img_hwc, dtype=np.float32) + 0.5, 0.0, 1.0).astype(np.float32, copy=False)


def numpy_test_image_alibi(idx: int, root: str, train: bool = False) -> np.ndarray:
    """Single image (32,32,3) float32 Alibi from torchvision CIFAR10."""
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=root, train=train, download=True)
    img_arr, _y = ds.data[int(idx)], ds.targets[int(idx)]
    x = (np.asarray(img_arr, dtype=np.float32) / 255.0).astype(np.float32)
    lo, hi = -0.5, 0.5
    xmin, xmax = float(x.min()), float(x.max())
    x = ((x - xmin) / (xmax - xmin + 1e-12)) * (hi - lo) + lo
    return np.ascontiguousarray(x, dtype=np.float32)
