#!/usr/bin/env python3
"""Blank-canvas Graco analysis for ImageNet-100 → PDF."""

from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ANALYSIS))

from common.run_analysis import run_blank_graco_analysis  # noqa: E402

if __name__ == "__main__":
    run_blank_graco_analysis(["--dataset", "imagenet100", *sys.argv[1:]])
