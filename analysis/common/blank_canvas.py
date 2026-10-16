"""Fixed blank starting images in Alibi [-0.5, 0.5] (dataset-native layout)."""

from __future__ import annotations

import numpy as np

ALIBI_LO = -0.5
ALIBI_HI = 0.5


def blank_mnist_hw() -> np.ndarray:
    """(28, 28) grayscale Alibi — black."""
    return np.full((28, 28), ALIBI_LO, dtype=np.float32)


def blank_cifar_hwc() -> np.ndarray:
    """(32, 32, 3) RGB Alibi — white."""
    return np.full((32, 32, 3), ALIBI_HI, dtype=np.float32)


def blank_imagenet_chw() -> np.ndarray:
    """(3, 299, 299) RGB Alibi — white."""
    return np.full((3, 299, 299), ALIBI_HI, dtype=np.float32)
