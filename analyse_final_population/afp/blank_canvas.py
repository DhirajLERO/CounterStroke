"""Fixed blank canvases (Alibi [-0.5, 0.5])."""

from __future__ import annotations

import numpy as np

from .graco_shared import load_blank_canvas_analysis


def blank_alibi_for_dataset(dataset: str) -> np.ndarray:
    mod = load_blank_canvas_analysis()
    blank_cifar_hwc = mod.blank_cifar_hwc
    blank_imagenet_chw = mod.blank_imagenet_chw
    blank_mnist_hw = mod.blank_mnist_hw

    if dataset == "mnist":
        return blank_mnist_hw()
    if dataset == "cifar10":
        return blank_cifar_hwc()
    if dataset == "imagenet100":
        return blank_imagenet_chw()
    raise ValueError(dataset)
