from __future__ import annotations

import torch
import torch.nn as nn


def model_predict_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Probabilities (N, num_classes); supports predict_proba, log-softmax, or raw logits."""
    if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
        return model.predict_proba(x)
    out = model(x)
    if out.dim() != 2:
        raise ValueError("Expected (N, C) logits or log-probabilities.")
    exp_out = torch.exp(out)
    s = exp_out.sum(dim=-1, keepdim=True)
    if (s - 1.0).abs().max() < 0.05:
        return exp_out / s.clamp(min=1e-12)
    return torch.softmax(out, dim=-1)
