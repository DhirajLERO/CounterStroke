"""Paths relative to GracoImage root."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GRACOIMAGE_ROOT = PACKAGE_ROOT.parent
CONFIG_PATH = PACKAGE_ROOT / "config" / "datasets.yaml"
OUTPUTS_ROOT = PACKAGE_ROOT / "outputs"
ANALYSIS_ROOT = GRACOIMAGE_ROOT / "analysis"
MNIST_DIR = GRACOIMAGE_ROOT / "MNIST"
CIFAR_DIR = GRACOIMAGE_ROOT / "Cifar10"
IMNET_DIR = GRACOIMAGE_ROOT / "Imagenet"
