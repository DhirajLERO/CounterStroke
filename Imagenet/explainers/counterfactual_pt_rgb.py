"""Gradient counterfactual — same API as MNIST ``CounterfactualPT`` (vendored under this package)."""

from __future__ import annotations

from .counterfactual_pt import CounterfactualExplanation, CounterfactualPT

__all__ = ["CounterfactualPT", "CounterfactualExplanation"]
