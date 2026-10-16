"""Classifier loading and softmax for blank-canvas belief views."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .blank_canvas import blank_alibi_for_dataset
from .dataset_imports import prepend_dataset_root, prepend_graco_dir
from .datasets import AppDatasetSpec


@lru_cache(maxsize=8)
def _load_model_cached(dataset: str, weights_str: str, device_str: str):
    prepend_dataset_root(dataset)
    weights = Path(weights_str)
    device = torch.device(device_str)

    if dataset == "mnist":
        from explainers.mnist_cnn import MNISTNet  # noqa: WPS433

        model = MNISTNet()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval().to(device)
        return model

    if dataset == "cifar10":
        from explainers.cifar_resnet import CifarResNet  # noqa: WPS433

        model = CifarResNet()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval().to(device)
        return model

    if dataset == "imagenet100":
        from explainers.imagenet100_inception import load_imagenet100_classifier  # noqa: WPS433

        return load_imagenet100_classifier(weights, device, num_classes=100)

    raise ValueError(dataset)


def predict_probs(
    spec: AppDatasetSpec,
    img_alibi: np.ndarray,
    weights: Path,
    device: torch.device,
) -> np.ndarray:
    model = _load_model_cached(spec.name, str(weights.resolve()), str(device))
    prepend_dataset_root(spec.name)

    if spec.name == "imagenet100":
        prepend_graco_dir(spec.graco_dir)
        from inception_infer import predict_probs_numpy_inception  # noqa: WPS433

        x = np.asarray(img_alibi, dtype=np.float32)[np.newaxis, ...]
        _, _, probs = predict_probs_numpy_inception(model, x, device=device)
        return probs[0]

    prepend_graco_dir(spec.graco_dir)
    from infer_utils import predict_probs_numpy  # noqa: WPS433

    x = np.asarray(img_alibi, dtype=np.float32)
    if spec.name == "mnist":
        # MNIST infer_utils: (N, H, W) — blank is (28, 28), no channel axis
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        if x.ndim != 3:
            raise ValueError(f"MNIST alibi must be (H,W) or (N,H,W), got {x.shape}")
    elif spec.name == "cifar10":
        # CIFAR infer_utils: (N, H, W, 3)
        if x.ndim == 3:
            x = x[np.newaxis, ...]
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"CIFAR alibi must be (H,W,3) or (N,H,W,3), got {x.shape}")
    else:
        raise ValueError(spec.name)

    _, _, probs = predict_probs_numpy(model, x, device=device)
    return probs[0]


def predict_blank(
    spec: AppDatasetSpec,
    weights: Path,
    device: torch.device,
) -> tuple[int, np.ndarray, np.ndarray]:
    blank = blank_alibi_for_dataset(spec.name)
    probs = predict_probs(spec, blank, weights, device)
    pred = int(np.argmax(probs))
    return pred, probs, blank


def alibi_hwc_for_classifier(spec: AppDatasetSpec, img_alibi: np.ndarray) -> np.ndarray:
    a = np.asarray(img_alibi, dtype=np.float32)
    if spec.name == "imagenet100":
        return a
    if a.ndim == 3 and a.shape[0] == 3:
        return np.transpose(a, (1, 2, 0))
    if a.ndim == 2:
        return a[:, :, np.newaxis]
    return a


def cf_01_to_alibi(spec: AppDatasetSpec, cf_01: np.ndarray) -> np.ndarray:
    if spec.name == "imagenet100":
        return np.transpose(cf_01.astype(np.float32) - 0.5, (2, 0, 1))
    a = cf_01.astype(np.float32)
    if a.ndim == 2:
        return a - 0.5
    return a - 0.5
