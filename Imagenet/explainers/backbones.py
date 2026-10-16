"""Frozen torchvision ResNet-18 backbone (features before classifier)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class FrozenResNet18Features(nn.Module):
    """Outputs 512-d pooled embedding from CHW inputs in Alibi [-0.5, 0.5]."""

    def __init__(self) -> None:
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.body = nn.Sequential(*list(m.children())[:-1])  # drop fc
        for p in self.body.parameters():
            p.requires_grad_(False)
        self.eval()

    @staticmethod
    def _to_imagenet_norm(x01: torch.Tensor) -> torch.Tensor:
        mean = x01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x01 - mean) / std

    def forward(self, x_alibi: torch.Tensor) -> torch.Tensor:
        """x_alibi: (N,3,H,W) in [-0.5, 0.5]."""
        x01 = (x_alibi + 0.5).clamp(0.0, 1.0)
        z = self.body(self._to_imagenet_norm(x01))
        return torch.flatten(z, 1)
