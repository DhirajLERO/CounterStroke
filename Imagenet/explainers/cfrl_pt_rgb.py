"""CFRL on RGB ImageNet tensors — training happens in ``cfrl_runner_rgb.train_cfrl_rgb_explainer``."""

from __future__ import annotations

from alibi.explainers import CounterfactualRL  # noqa: F401 — dependency surface for callers

__all__ = ["CounterfactualRL"]
