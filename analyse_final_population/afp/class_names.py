"""Human-readable class names per dataset."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .graco_shared import load_val_by_class


@lru_cache(maxsize=4)
def get_class_names(dataset: str, class_map_str: str | None = None) -> list[str]:
    mod = load_val_by_class()
    class_names_cifar = mod.class_names_cifar
    class_names_imagenet = mod.class_names_imagenet
    class_names_mnist = mod.class_names_mnist

    if dataset == "mnist":
        return class_names_mnist()
    if dataset == "cifar10":
        return class_names_cifar()
    if dataset == "imagenet100":
        cmap = Path(class_map_str) if class_map_str else None
        return class_names_imagenet(cmap)
    raise ValueError(dataset)


def label_name(names: list[str], cid: int) -> str:
    if 0 <= cid < len(names):
        return names[cid]
    return str(cid)
