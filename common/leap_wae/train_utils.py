"""Training helpers ported from LeapFactual ``experiments/utils.py`` (no PyTorch Lightning)."""

from __future__ import annotations

import numpy as np


def frange_cycle_linear(start: float, stop: float, n_epoch: int, n_cycle: int = 1, ratio: float = 0.5) -> np.ndarray:
    """Linear GAN / ADV weight schedule (same as official LeapFactual ExpGAN)."""
    L = np.ones(n_epoch, dtype=np.float64) * stop
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)
    for c in range(n_cycle):
        v, i = start, 0.0
        while v <= stop and int(i + c * period) < n_epoch:
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L
