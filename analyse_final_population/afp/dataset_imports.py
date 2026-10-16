"""Import dataset-local packages (``explainers``, ``infer_utils``, …) without cache clashes."""

from __future__ import annotations

import sys
from pathlib import Path

from .paths import CIFAR_DIR, IMNET_DIR, MNIST_DIR

_DATASET_ROOTS: dict[str, Path] = {
    "mnist": MNIST_DIR,
    "cifar10": CIFAR_DIR,
    "imagenet100": IMNET_DIR,
}


def dataset_root(dataset: str) -> Path:
    try:
        return _DATASET_ROOTS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset}") from exc


def _evict_explainers_cache(expected_root: str) -> None:
    mod = sys.modules.get("explainers")
    if mod is None:
        return
    paths = getattr(mod, "__path__", None)
    if paths:
        loc = str(paths[0])
    else:
        loc = str(getattr(mod, "__file__", "") or "")
    if expected_root in loc:
        return
    for key in list(sys.modules):
        if key == "explainers" or key.startswith("explainers."):
            del sys.modules[key]


def prepend_dataset_root(dataset: str) -> Path:
    """Put ``MNIST`` / ``Cifar10`` / ``Imagenet`` first on ``sys.path`` for ``explainers.*`` imports."""
    root = dataset_root(dataset)
    root_s = str(root.resolve())
    if root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)
    _evict_explainers_cache(root_s)
    return root


_GRACO_INFER_MODULES = ("infer_utils", "inception_infer")


def _evict_graco_infer_cache(expected_graco: str) -> None:
    for name in _GRACO_INFER_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        loc = str(getattr(mod, "__file__", "") or "")
        if expected_graco in loc:
            continue
        del sys.modules[name]


def prepend_graco_dir(graco_dir: Path) -> Path:
    """Put the dataset ``graco/`` dir first and reload ``infer_utils`` / ``inception_infer`` if needed."""
    graco_s = str(graco_dir.resolve())
    if graco_s in sys.path:
        sys.path.remove(graco_s)
    sys.path.insert(0, graco_s)
    _evict_graco_infer_cache(graco_s)
    return graco_dir
