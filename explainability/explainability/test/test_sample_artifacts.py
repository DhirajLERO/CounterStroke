"""Sample artifact completeness checks for batch resume."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explainability.io.artifacts import (  # noqa: E402
    sample_artifacts_complete,
    sample_output_dir,
)


def test_sample_artifacts_complete_detects_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "explainability.paths.OUTPUTS_ROOT",
        tmp_path,
    )
    dataset = "mnist"
    sample_idx = 7
    methods = ["change_frequency", "gradcam"]
    out_dir = sample_output_dir(dataset, sample_idx)

    complete, missing = sample_artifacts_complete(
        dataset,
        sample_idx,
        methods,
        save_overlays=True,
        save_srg_plots=True,
    )
    assert not complete
    assert "meta.json" in missing

    out_dir.mkdir(parents=True)
    (out_dir / "meta.json").write_text(json.dumps({"sample_idx": sample_idx}), encoding="utf-8")
    (out_dir / "maps").mkdir()
    (out_dir / "srg").mkdir()
    (out_dir / "overlays").mkdir()
    (out_dir / "srg_plots").mkdir()

    for method in methods:
        np.savez_compressed(out_dir / "maps" / f"{method}.npz", importance=np.zeros((2, 2)))
        np.savez_compressed(out_dir / "srg" / f"{method}.npz", srg_score=np.array(0.1))
        (out_dir / "overlays" / f"{method}.png").write_bytes(b"png")
        (out_dir / "srg_plots" / f"{method}.png").write_bytes(b"png")

    complete, missing = sample_artifacts_complete(
        dataset,
        sample_idx,
        methods,
        save_overlays=True,
        save_srg_plots=True,
    )
    assert not complete
    assert "srg_plots/_all_methods.png" in missing

    (out_dir / "srg_plots" / "_all_methods.png").write_bytes(b"png")
    complete, missing = sample_artifacts_complete(
        dataset,
        sample_idx,
        methods,
        save_overlays=True,
        save_srg_plots=True,
    )
    assert complete
    assert missing == []
