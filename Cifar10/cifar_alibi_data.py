"""CIFAR-10 in Alibi scale [-0.5, 0.5] NHWC — torchvision only."""

from __future__ import annotations

import numpy as np
from torchvision.datasets import CIFAR10


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def load_cifar10_alibi(root: str | None = None) -> tuple[tuple, tuple]:
    """Min/max per split maps each split's pixels to [-0.5, 0.5] (matches MNIST helper style)."""
    root = root or "."
    train_raw = CIFAR10(root=root, train=True, download=True)
    test_raw = CIFAR10(root=root, train=False, download=True)

    def pack(ds):
        x = np.asarray(ds.data, dtype=np.float32) / 255.0
        y = np.asarray(ds.targets, dtype=np.int64)
        lo, hi = -0.5, 0.5
        xmin, xmax = float(x.min()), float(x.max())
        x = ((x - xmin) / (xmax - xmin + 1e-12)) * (hi - lo) + lo
        return x.astype(np.float32), y, _one_hot(y, 10)

    x_train, y_train, y_train_ohe = pack(train_raw)
    x_test, y_test, y_test_ohe = pack(test_raw)
    return (x_train, y_train, y_train_ohe), (x_test, y_test, y_test_ohe)


def nchw_to_nhwc(x: np.ndarray) -> np.ndarray:
    return np.transpose(x.astype(np.float32), (0, 2, 3, 1))


def nhwc_to_nchw(x: np.ndarray) -> np.ndarray:
    return np.transpose(x.astype(np.float32), (0, 3, 1, 2))


def alibi_nhwc_to_cfrl01_nchw(x_nhwc: np.ndarray) -> np.ndarray:
    x01 = np.clip(x_nhwc.astype(np.float32) + 0.5, 0.0, 1.0)
    return nhwc_to_nchw(x01)
