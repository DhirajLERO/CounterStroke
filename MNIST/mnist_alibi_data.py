"""MNIST in Alibi scale [-0.5, 0.5] NHWC — torchvision only (no TensorFlow)."""

from __future__ import annotations

import numpy as np
from torchvision.datasets import MNIST


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def load_mnist_alibi(mnist_root: str | None = None) -> tuple[tuple, tuple]:
    """Match keras cf_mnist: [0,1] then per-split min/max map to [-0.5, 0.5]."""
    root = mnist_root or "."
    train_raw = MNIST(root=root, train=True, download=True, transform=None)
    test_raw = MNIST(root=root, train=False, download=True, transform=None)

    def pack(ds):
        x = (ds.data.float().numpy() / 255.0).reshape(-1, 28, 28, 1).astype(np.float32)
        y = ds.targets.numpy().astype(np.int64)
        lo, hi = -0.5, 0.5
        xmin, xmax = float(x.min()), float(x.max())
        x = ((x - xmin) / (xmax - xmin + 1e-12)) * (hi - lo) + lo
        return x, y, _one_hot(y, 10)

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


def np_mnist_to_nchw(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        return x
    if x.shape[1] == 1 and x.shape[2] == 28 and x.shape[3] == 28:
        return x
    if x.shape[1] == 28 and x.shape[2] == 28 and x.shape[3] == 1:
        return nhwc_to_nchw(x)
    raise ValueError(f"Expected MNIST NHWC or NCHW, got {x.shape}")
