"""Load dataset task classifiers (read-only imports from GracoImage packages)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .paths import CIFAR_DIR, IMNET_DIR, MNIST_DIR


def load_mnist_classifier(weights: Path, device: torch.device):
    sys.path.insert(0, str(MNIST_DIR))
    from explainers.mnist_cnn import MNISTNet  # noqa: WPS433

    model = MNISTNet()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval().to(device)
    return model


def load_cifar_classifier(weights: Path, device: torch.device):
    sys.path.insert(0, str(CIFAR_DIR))
    from explainers.cifar_resnet import CifarResNet  # noqa: WPS433

    model = CifarResNet()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval().to(device)
    return model


def load_imagenet_classifier(weights: Path, device: torch.device, num_classes: int = 100):
    sys.path.insert(0, str(IMNET_DIR))
    from explainers.imagenet100_inception import load_imagenet100_classifier  # noqa: WPS433

    return load_imagenet100_classifier(weights, device, num_classes=num_classes)
