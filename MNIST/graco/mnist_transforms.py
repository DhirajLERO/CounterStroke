from __future__ import annotations

import numpy as np
from torchvision import transforms

MNIST_MEAN: float = 0.1307
MNIST_STD: float = 0.3081


def mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def alibi_mnist_transform_hw() -> transforms.Compose:
    """MNIST as (28,28) float32 in [-0.5, 0.5], matching CF100 / ToTensor then minus 0.5."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t.squeeze(0) - 0.5).float()),
        ]
    )


def alibi_to_01(img_hw: np.ndarray) -> np.ndarray:
    """Map Alibi-range (H,W) float to [0, 1] for visualization and NPZ export."""
    return np.clip(np.asarray(img_hw, dtype=np.float32) + 0.5, 0.0, 1.0).astype(
        np.float32, copy=False
    )


def unnorm_to_01(img_norm_2d: np.ndarray) -> np.ndarray:
    """
    Converts a normalized MNIST image (H,W) back to float32 in [0,1].
    """
    out = img_norm_2d * MNIST_STD + MNIST_MEAN
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

