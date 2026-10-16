"""Paths relative to GracoImage root."""

from __future__ import annotations

from pathlib import Path

GRACOIMAGE_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = GRACOIMAGE_ROOT / "analysis"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MNIST_DIR = GRACOIMAGE_ROOT / "MNIST"
CIFAR_DIR = GRACOIMAGE_ROOT / "Cifar10"
CIFAR_GRACO_DIR = CIFAR_DIR / "graco"
IMNET_DIR = GRACOIMAGE_ROOT / "Imagenet"
