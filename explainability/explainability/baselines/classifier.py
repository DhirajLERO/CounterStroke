"""Classifier loading and inference for explainability baselines."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import DatasetSpec
from ..imports_helper import prepend_dataset_root, prepend_graco_dir


@lru_cache(maxsize=8)
def load_model(dataset: str, weights_str: str, device_str: str) -> nn.Module:
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


def alibi_to_model_batch(spec: DatasetSpec, img_alibi: np.ndarray) -> torch.Tensor:
    x = np.asarray(img_alibi, dtype=np.float32)
    if spec.name == "imagenet100":
        if x.ndim == 3:
            t = torch.from_numpy(x[np.newaxis, ...])
        else:
            t = torch.from_numpy(x)
        return t
    if spec.name == "mnist":
        if x.ndim == 2:
            t = torch.from_numpy(x[np.newaxis, np.newaxis, ...])
        elif x.ndim == 3 and x.shape[-1] == 1:
            t = torch.from_numpy(x[..., 0][np.newaxis, np.newaxis, ...])
        elif x.ndim == 3:
            t = torch.from_numpy(x[np.newaxis, ...])
        else:
            t = torch.from_numpy(x)
        return t
    if spec.name == "cifar10":
        if x.ndim == 3:
            arr = np.transpose(x, (2, 0, 1))[np.newaxis, ...]
        else:
            arr = np.transpose(x, (0, 3, 1, 2))
        return torch.from_numpy(arr)
    raise ValueError(spec.name)


def predict_argmax_class(
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    weights: Path,
    device: torch.device,
) -> int:
    probs = predict_probs(spec, img_alibi, weights, device)
    return int(np.argmax(probs))


def _stack_alibi_batch(spec: DatasetSpec, images_alibi: list[np.ndarray]) -> np.ndarray:
    """Stack per-image Alibi arrays into a batched numpy array for inference."""
    if not images_alibi:
        raise ValueError("images_alibi must be non-empty")
    arrs = [np.asarray(img, dtype=np.float32) for img in images_alibi]
    if spec.name == "imagenet100":
        return np.stack(arrs, axis=0)
    if spec.name == "mnist":
        flat: list[np.ndarray] = []
        for x in arrs:
            if x.ndim == 3 and x.shape[-1] == 1:
                flat.append(x[..., 0])
            elif x.ndim == 2:
                flat.append(x)
            else:
                flat.append(x.reshape(x.shape[-2], x.shape[-1]))
        return np.stack(flat, axis=0)
    if spec.name == "cifar10":
        return np.stack(arrs, axis=0)
    raise ValueError(spec.name)


def predict_class_probs_batch(
    spec: DatasetSpec,
    images_alibi: list[np.ndarray],
    target_class: int,
    weights: Path,
    device: torch.device,
    *,
    batch_size: int = 32,
) -> list[float]:
    """Return ``p(target_class)`` for each image via batched classifier inference."""
    if target_class < 0:
        raise ValueError(f"Invalid target_class={target_class}; class index was not resolved")
    if not images_alibi:
        return []

    model = load_model(spec.name, str(weights.resolve()), str(device))
    prepend_dataset_root(spec.name)
    x = _stack_alibi_batch(spec, images_alibi)

    if spec.name == "imagenet100":
        prepend_graco_dir(spec.graco_dir)
        from inception_infer import predict_probs_numpy_inception  # noqa: WPS433

        _, _, probs = predict_probs_numpy_inception(
            model, x, device=device, batch_size=int(batch_size)
        )
    else:
        prepend_graco_dir(spec.graco_dir)
        from infer_utils import predict_probs_numpy  # noqa: WPS433

        _, _, probs = predict_probs_numpy(model, x, device=device, batch_size=int(batch_size))

    if target_class >= probs.shape[1]:
        raise ValueError(f"target_class={target_class} out of range for {probs.shape[1]} classes")
    return [float(probs[i, target_class]) for i in range(probs.shape[0])]


def predict_class_prob(
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    target_class: int,
    weights: Path,
    device: torch.device,
) -> float:
    return predict_class_probs_batch(
        spec, [img_alibi], target_class, weights, device, batch_size=1
    )[0]


def predict_probs(
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    weights: Path,
    device: torch.device,
) -> np.ndarray:
    model = load_model(spec.name, str(weights.resolve()), str(device))
    prepend_dataset_root(spec.name)

    if spec.name == "imagenet100":
        prepend_graco_dir(spec.graco_dir)
        from inception_infer import predict_probs_numpy_inception  # noqa: WPS433

        x = np.asarray(img_alibi, dtype=np.float32)
        if x.ndim == 3:
            x = x[np.newaxis, ...]
        _, _, probs = predict_probs_numpy_inception(model, x, device=device)
        return probs[0]

    prepend_graco_dir(spec.graco_dir)
    from infer_utils import predict_probs_numpy  # noqa: WPS433

    x = np.asarray(img_alibi, dtype=np.float32)
    if spec.name == "mnist":
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
        if x.ndim == 2:
            x = x[np.newaxis, ...]
    elif spec.name == "cifar10":
        if x.ndim == 3:
            x = x[np.newaxis, ...]
    _, _, probs = predict_probs_numpy(model, x, device=device)
    return probs[0]


def gradcam_target_layer(spec: DatasetSpec, model: nn.Module) -> nn.Module:
    if spec.name == "mnist":
        return model.conv2
    if spec.name == "cifar10":
        return model.backbone.layer4
    if spec.name == "imagenet100":
        return model.net.Mixed_7c
    raise ValueError(spec.name)


def channel_max_saliency(attribution: np.ndarray) -> np.ndarray:
    """Legacy helper; prefer attribution_to_spatial_map with the reference image."""
    arr = np.asarray(attribution, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[1] == arr.shape[2]:
            return np.abs(arr).max(axis=0)
        if arr.shape[-1] in (1, 3):
            return np.abs(arr).max(axis=-1)
        return np.abs(arr).max(axis=0)
    return np.abs(arr)
