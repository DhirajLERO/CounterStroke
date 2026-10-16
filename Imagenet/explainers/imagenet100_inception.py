"""Fine-tuned InceptionV3 classifier for ImageNet-100 (100-way, Alibi inputs)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNet100Inception(nn.Module):
    """Log-softmax outputs (Alibi CHW [-0.5, 0.5] in, ImageNet normalize internally)."""

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

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._norm(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward_logits(x), dim=1)


def build_imagenet100_inception(num_classes: int = 100) -> ImageNet100Inception:
    from torchvision import models

    m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    m.aux_logits = False
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, int(num_classes))
    return ImageNet100Inception(m)


def load_imagenet100_classifier(
    weights_path: Path | str,
    device: torch.device,
    *,
    num_classes: int = 100,
) -> ImageNet100Inception:
    model = build_imagenet100_inception(num_classes=num_classes)
    state = torch.load(Path(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
