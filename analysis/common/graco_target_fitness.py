"""Target-class O1: p(label) - p(wanted_class) (minimize to favor wanted_class)."""

from __future__ import annotations

import numpy as np
import torch


def resolve_inference_device(model, device: str | torch.device | None) -> torch.device:
    """Use the model's device when config omits ``device`` or disagrees with weights."""
    try:
        model_dev = next(model.parameters()).device
    except StopIteration:
        model_dev = torch.device("cpu")
    if device is None:
        return model_dev
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if model_dev.type == "cuda" and dev.type != "cuda":
        return model_dev
    return dev


def score_o1(probs: np.ndarray, label: int, wanted_class: int) -> np.ndarray:
    """Batch scores shape (N,)."""
    return probs[:, int(label)] - probs[:, int(wanted_class)]


def evaluate_probability_target(
    individuals_np: np.ndarray,
    label: int,
    wanted_class: int,
    model_report_fn,
    model,
    batch_size: int,
    device,
    amp: bool = False,
) -> list[float]:
    if isinstance(individuals_np, list):
        individuals_np = np.stack(individuals_np, axis=0)
    dev = resolve_inference_device(model, device)
    _, _, probs = model_report_fn(
        model, individuals_np, device=dev, batch_size=batch_size, amp=amp
    )
    return score_o1(probs, label, wanted_class).tolist()
