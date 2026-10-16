"""Load eval_manifest.json entries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    sample_idx: int
    y_true: int
    split: str
    folder_idx: int


def load_eval_manifest(manifest_path: Path) -> list[ManifestEntry]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Manifest must be a JSON list: {manifest_path}")
    out: list[ManifestEntry] = []
    for i, rec in enumerate(raw):
        idx = int(rec.get("sample_idx", rec.get("dataset_index", -1)))
        if idx < 0:
            raise ValueError(f"Entry {i} missing sample_idx: {rec!r}")
        y = int(rec["y_true"])
        out.append(
            ManifestEntry(
                sample_idx=idx,
                y_true=y,
                split=str(rec.get("split", "test")),
                folder_idx=i,
            )
        )
    return out
