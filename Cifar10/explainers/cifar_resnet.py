"""ResNet with CIFAR-style stem (3×3 stride 1, no ImageNet maxpool) for 32×32 RGB."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CifarResNet(nn.Module):
    """Log-softmax outputs (matches MNISTNet interface)."""

    def __init__(self, num_classes: int = 10, depth: int = 18) -> None:
        super().__init__()
        if depth != 18:
            raise ValueError("Only depth=18 wired; extend factory for ResNet34/50.")
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.backbone = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.backbone(x), dim=1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward_logits(x), dim=1)


def conv1x1_trunk_features(model: CifarResNet, x: torch.Tensor) -> torch.Tensor:
    """Frozen-feature extractor: global pool before classifier (output dim = 512 for ResNet-18)."""
    m = model.backbone
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.layer1(m.maxpool(x))
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)
    x = m.avgpool(x)
    return torch.flatten(x, 1)


def build_cifar_resnet(num_classes: int = 10, depth: int = 18) -> CifarResNet:
    return CifarResNet(num_classes=num_classes, depth=depth)
