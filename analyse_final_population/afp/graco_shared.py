"""Import helpers for GracoImage/analysis/common (avoids clash with GracoImage/common)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parents[2] / "analysis"


def ensure_analysis_path() -> None:
    p = str(_ANALYSIS)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_graco_mine():
    ensure_analysis_path()
    return importlib.import_module("common.graco_mine")


def load_graco_blank_evolve():
    ensure_analysis_path()
    return importlib.import_module("common.graco_blank_evolve")


def load_classifier_load():
    ensure_analysis_path()
    return importlib.import_module("common.classifier_load")


def load_val_by_class():
    ensure_analysis_path()
    return importlib.import_module("common.val_by_class")


def load_blank_canvas_analysis():
    ensure_analysis_path()
    return importlib.import_module("common.blank_canvas")


def load_ssim():
    ensure_analysis_path()
    return importlib.import_module("common.ssim")
