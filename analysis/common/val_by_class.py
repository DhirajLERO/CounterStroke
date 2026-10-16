"""Load val/test images grouped by class (display RGB/gray in [0, 1])."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .paths import CIFAR_DIR, GRACOIMAGE_ROOT, IMNET_DIR, MNIST_DIR


def _alibi_hwc_to_01(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 3 and a.shape[-1] in (1, 3):
        return np.clip(a + 0.5, 0.0, 1.0)
    if a.ndim == 3 and a.shape[0] == 3:
        return np.clip(np.transpose(a, (1, 2, 0)) + 0.5, 0.0, 1.0)
    if a.ndim == 2:
        return np.clip(a + 0.5, 0.0, 1.0)
    raise ValueError(f"Unsupported shape {a.shape}")


def val_pool_mnist(data_root: Path) -> dict[int, list[np.ndarray]]:
    sys.path.insert(0, str(MNIST_DIR))
    from mnist_alibi_data import load_mnist_alibi  # noqa: WPS433

    (_tr, _ytr, _), (x_te, y_te, _) = load_mnist_alibi(str(data_root))
    by: dict[int, list[np.ndarray]] = defaultdict(list)
    for i in range(len(y_te)):
        img = x_te[i]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[:, :, 0]
        by[int(y_te[i])].append(_alibi_hwc_to_01(img))
    return dict(by)


def val_pool_cifar(data_root: Path) -> dict[int, list[np.ndarray]]:
    sys.path.insert(0, str(CIFAR_DIR))
    from cifar_alibi_data import load_cifar10_alibi  # noqa: WPS433

    (_tr, _ytr, _), (x_te, y_te, _) = load_cifar10_alibi(str(data_root))
    by: dict[int, list[np.ndarray]] = defaultdict(list)
    for i in range(len(y_te)):
        by[int(y_te[i])].append(_alibi_hwc_to_01(x_te[i]))
    return dict(by)


def val_pool_imagenet(data_root: Path, class_map_path: Path | None = None) -> dict[int, list[np.ndarray]]:
    sys.path.insert(0, str(IMNET_DIR))
    from imagenet100_classmap import resolve_class_map_path  # noqa: WPS433
    from imagenet100_dataset import ImageNet100AlibiDataset  # noqa: WPS433

    cmap_p = resolve_class_map_path(class_map_path)
    ds = ImageNet100AlibiDataset(data_root, "val", class_map_path=cmap_p)
    by: dict[int, list[np.ndarray]] = defaultdict(list)
    for i in range(len(ds)):
        img_chw, y = ds[i]
        arr = img_chw.numpy()
        by[int(y)].append(_alibi_hwc_to_01(arr))
    return dict(by)


def class_names_mnist() -> list[str]:
    return [str(i) for i in range(10)]


def class_names_cifar() -> list[str]:
    """Torchvision's CIFAR10 has historically exposed class names either as"""
    try:
        from torchvision.datasets import CIFAR10  # noqa: WPS433

        # Many versions expose `.classes` on the dataset instance.
        try:
            ds = CIFAR10(root=".", train=False, download=False)
            if hasattr(ds, "classes") and ds.classes:  # type: ignore[attr-defined]
                return list(ds.classes)  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: canonical CIFAR-10 label names (stable).
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]


def class_names_imagenet(class_map_path: Path | None = None) -> list[str]:
    sys.path.insert(0, str(IMNET_DIR))
    from imagenet100_classmap import load_class_map, resolve_class_map_path  # noqa: WPS433

    cmap = load_class_map(resolve_class_map_path(class_map_path))
    return list(cmap["class_names"])
