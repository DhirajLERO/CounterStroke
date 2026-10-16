"""Standard XAI baselines on the factual original image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import DatasetSpec
from ..phenotype.aggregations import normalize_01
from ..phenotype.diffs import attribution_to_spatial_map
from .classifier import (
    alibi_to_model_batch,
    gradcam_target_layer,
    load_model,
)


def _saliency_hw(spec: DatasetSpec, img_alibi: np.ndarray, arr: np.ndarray) -> np.ndarray:
    sal = attribution_to_spatial_map(img_alibi, arr)
    return normalize_01(sal)


def integrated_gradients(
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    target_class: int,
    weights: Path,
    device: torch.device,
    *,
    steps: int = 32,
) -> np.ndarray:
    from captum.attr import IntegratedGradients

    model = load_model(spec.name, str(weights.resolve()), str(device))
    model.eval()
    x = alibi_to_model_batch(spec, img_alibi).to(device).requires_grad_(True)
    baseline = torch.zeros_like(x)

    def forward_func(inp: torch.Tensor) -> torch.Tensor:
        return _target_forward(model, target_class)(inp)

    ig = IntegratedGradients(forward_func)
    attr = ig.attribute(x, baselines=baseline, n_steps=steps)
    return _saliency_hw(spec, img_alibi, attr.detach().cpu().numpy())


def _target_forward(model, target_class: int):
    def forward_func(inp: torch.Tensor) -> torch.Tensor:
        logits = model.forward_logits(inp) if hasattr(model, "forward_logits") else model(inp)
        return logits[:, target_class]

    return forward_func


def input_times_gradient(
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    target_class: int,
    weights: Path,
    device: torch.device,
) -> np.ndarray:
    from captum.attr import InputXGradient

    model = load_model(spec.name, str(weights.resolve()), str(device))
    model.eval()
    x = alibi_to_model_batch(spec, img_alibi).to(device).requires_grad_(True)
    attr = InputXGradient(_target_forward(model, target_class)).attribute(x)
    return _saliency_hw(spec, img_alibi, attr.detach().cpu().numpy())


def smoothgrad(
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    target_class: int,
    weights: Path,
    device: torch.device,
    *,
    n_samples: int = 15,
    stdev: float = 0.02,
) -> np.ndarray:
    from captum.attr import Saliency

    model = load_model(spec.name, str(weights.resolve()), str(device))
    model.eval()
    x = alibi_to_model_batch(spec, img_alibi).to(device).requires_grad_(True)
    sal = Saliency(_target_forward(model, target_class))
    attrs: list[torch.Tensor] = []
    for _ in range(n_samples):
        noise = torch.randn_like(x) * stdev
        attrs.append(sal.attribute(x + noise))
    attr = torch.stack(attrs, dim=0).mean(dim=0)
    return _saliency_hw(spec, img_alibi, attr.detach().cpu().numpy())


def occlusion(
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    target_class: int,
    weights: Path,
    device: torch.device,
    *,
    window: int = 3,
    stride: int = 1,
) -> np.ndarray:
    from captum.attr import Occlusion

    model = load_model(spec.name, str(weights.resolve()), str(device))
    model.eval()
    x = alibi_to_model_batch(spec, img_alibi).to(device).requires_grad_(True)
    window_shape = (1, window, window)
    occ = Occlusion(_target_forward(model, target_class))
    attr = occ.attribute(
        x,
        sliding_window_shapes=window_shape,
        strides=(1, stride, stride),
        baselines=0.0,
    )
    return _saliency_hw(spec, img_alibi, attr.detach().cpu().numpy())


def gradcam(
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    target_class: int,
    weights: Path,
    device: torch.device,
) -> np.ndarray:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model = load_model(spec.name, str(weights.resolve()), str(device))
    model.eval()
    x_np = alibi_to_model_batch(spec, img_alibi).numpy()
    target_layers = [gradcam_target_layer(spec, model)]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale = cam(input_tensor=torch.from_numpy(x_np).to(device), targets=targets)[0, :]
    return normalize_01(grayscale.astype(np.float32))


def shap_gradient(
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    target_class: int,
    weights: Path,
    device: torch.device,
    *,
    background: np.ndarray | None = None,
) -> np.ndarray:
    import shap
    import torch.nn as nn

    model = load_model(spec.name, str(weights.resolve()), str(device))
    model.eval()
    x = alibi_to_model_batch(spec, img_alibi).to(device)

    if background is None:
        bg = torch.zeros_like(x)
    else:
        bg = alibi_to_model_batch(spec, background).to(device)

    class _TargetLogitModel(nn.Module):
        def __init__(self, inner: nn.Module, cls: int) -> None:
            super().__init__()
            self.inner = inner
            self.cls = cls

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            logits = self.inner.forward_logits(inp) if hasattr(self.inner, "forward_logits") else self.inner(inp)
            return logits[:, self.cls : self.cls + 1]

    wrapped = _TargetLogitModel(model, target_class)
    explainer = shap.GradientExplainer(wrapped, bg)
    shap_values = explainer.shap_values(x)
    if isinstance(shap_values, list):
        arr = shap_values[0]
    else:
        arr = shap_values
    return _saliency_hw(spec, img_alibi, np.asarray(arr, dtype=np.float32))


BASELINE_METHODS = {
    "integrated_gradients": integrated_gradients,
    "input_x_gradient": input_times_gradient,
    "smoothgrad": smoothgrad,
    "occlusion": occlusion,
    "gradcam": gradcam,
    "shap_gradient": shap_gradient,
}


def compute_baseline(
    method: str,
    spec: DatasetSpec,
    img_alibi: np.ndarray,
    target_class: int,
    weights: Path,
    device: torch.device,
) -> np.ndarray:
    fn = BASELINE_METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown baseline method: {method}")
    return fn(spec, img_alibi, target_class, weights, device)
