"""Inception v3 with inputs in Alibi CHW [-0.5, 0.5]."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InceptionAlibiMNISTStyle(nn.Module):
    """Log-softmax output like MNISTNet for CounterfactualPT."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        x01 = (x + 0.5).clamp(0.0, 1.0)
        mean = x01.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x01.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (x01 - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.net(self._norm(x)), dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(self._norm(x)), dim=1)


def load_inception_alibi(device: torch.device) -> InceptionAlibiMNISTStyle:
    from torchvision import models

    m = models.inception_v3(weights="IMAGENET1K_V1")
    m.aux_logits = False
    m.eval()
    w = InceptionAlibiMNISTStyle(m).to(device)
    for p in w.parameters():
        p.requires_grad_(False)
    return w
