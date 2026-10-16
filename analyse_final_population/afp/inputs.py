"""Load factual eval images in Alibi space and display [0, 1]."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .datasets import AppDatasetSpec
from .manifest import ManifestEntry
from .paths import CIFAR_GRACO_DIR, IMNET_DIR, MNIST_DIR


def alibi_to_display_01(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 3:
        return np.clip(np.transpose(a, (1, 2, 0)) + 0.5, 0.0, 1.0)
    if a.ndim == 3 and a.shape[-1] == 1:
        g = np.clip(a[:, :, 0] + 0.5, 0.0, 1.0)
        return g
    if a.ndim == 2:
        return np.clip(a + 0.5, 0.0, 1.0)
    if a.ndim == 3:
        return np.clip(a + 0.5, 0.0, 1.0)
    raise ValueError(f"Unsupported image shape {a.shape}")


def display_to_rgb_uint8(img_01: np.ndarray) -> np.ndarray:
    d = np.asarray(img_01, dtype=np.float32)
    if d.ndim == 2:
        g = (np.clip(d, 0, 1) * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    rgb = (np.clip(d, 0, 1) * 255).astype(np.uint8)
    if rgb.shape[-1] == 1:
        g = rgb[:, :, 0]
        return np.stack([g, g, g], axis=-1)
    return rgb


def load_factual_alibi(spec: AppDatasetSpec, entry: ManifestEntry, data_root: Path) -> np.ndarray:
    if spec.name == "mnist":
        sys.path.insert(0, str(MNIST_DIR))
        from torchvision.datasets import MNIST  # noqa: WPS433

        ds = MNIST(root=str(data_root), train=False, download=False, transform=None)
        img_t, _ = ds[int(entry.sample_idx)]
        x = (np.asarray(img_t, dtype=np.float32) / 255.0).reshape(28, 28).astype(np.float32) - 0.5
        return x

    if spec.name == "cifar10":
        sys.path.insert(0, str(CIFAR_GRACO_DIR))
        from cifar_transforms import numpy_test_image_alibi  # noqa: WPS433

        return numpy_test_image_alibi(int(entry.sample_idx), str(data_root), train=False)

    if spec.name == "imagenet100":
        sys.path.insert(0, str(IMNET_DIR))
        from graco.imagenet_subset import load_imagenet_val_manifest_samples  # noqa: WPS433

        rows = load_imagenet_val_manifest_samples(
            data_root,
            [{"sample_idx": entry.sample_idx, "y_true": entry.y_true, "split": entry.split}],
            max_samples=1,
        )
        if not rows:
            raise FileNotFoundError(f"No ImageNet sample at index {entry.sample_idx}")
        _did, _yt, arr = rows[0]
        return np.asarray(arr, dtype=np.float32)

    raise ValueError(spec.name)
