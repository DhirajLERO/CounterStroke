"""Shared CFRL manifest target handling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def extract_intended_targets(
    manifest: Sequence[Mapping[str, Any]],
    *,
    num_classes: int,
) -> np.ndarray:
    """Return validated ``y_intended_target`` values as an int64 array."""
    if not manifest:
        raise ValueError("CFRL manifest must contain at least one entry")

    targets: list[int] = []
    for index, entry in enumerate(manifest):
        if "y_intended_target" not in entry:
            raise ValueError(
                f"Manifest entry {index} is missing required field 'y_intended_target'"
            )
        raw_target = entry["y_intended_target"]
        if isinstance(raw_target, bool) or not isinstance(raw_target, int):
            raise ValueError(
                f"Manifest entry {index} has non-integer y_intended_target: {raw_target!r}"
            )

        target = int(raw_target)
        if not 0 <= target < num_classes:
            raise ValueError(
                f"Manifest entry {index} has y_intended_target={target}; "
                f"expected a class in [0, {num_classes})"
            )
        if "y_true" in entry and target == int(entry["y_true"]):
            raise ValueError(
                f"Manifest entry {index} has y_intended_target equal to y_true ({target})"
            )
        targets.append(target)

    return np.asarray(targets, dtype=np.int64)
