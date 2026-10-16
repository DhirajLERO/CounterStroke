"""SRG batched inference path tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explainability.config import get_dataset_spec  # noqa: E402
from explainability.srg.evaluate import compute_srg  # noqa: E402


def test_batched_srg_scatters_probs():
    spec = get_dataset_spec("mnist")
    img = np.zeros((28, 28), dtype=np.float32)
    importance = np.arange(28 * 28, dtype=np.float32).reshape(28, 28)
    weights = spec.weights
    device = torch.device("cpu")

    calls: list[list[np.ndarray]] = []

    def _fake_batch(spec, images, target_class, weights, device, *, batch_size=32):
        calls.append(images)
        return [0.1 * (i + 1) for i in range(len(images))]

    with patch(
        "explainability.srg.evaluate.predict_class_prob",
        return_value=0.5,
    ), patch(
        "explainability.srg.evaluate.predict_class_probs_batch",
        side_effect=_fake_batch,
    ):
        result = compute_srg(
            spec,
            img,
            importance,
            prob_class=0,
            weights=weights,
            device=device,
            curve_points=3,
            infer_batch_size=16,
        )

    assert result.p_baseline == 0.5
    assert len(calls) == 1
    # curve_points=3 on 784 pixels -> 4 k values (0 + 3 positive); k=0 uses p0; 3 k>0 -> 6 images
    assert len(calls[0]) == 6
    assert result.mif_curve.shape == result.lif_curve.shape == (4,)
