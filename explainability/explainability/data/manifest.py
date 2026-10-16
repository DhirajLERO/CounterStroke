"""Iterate eval manifest entries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..config import DatasetSpec


@dataclass(frozen=True)
class ManifestEntry:
    sample_idx: int
    y_true: int
    y_intended_target: int | None
    raw: dict


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Eval manifest not found: {path}\n"
            "Batch mode reads eval_manifest.json (run_sample uses --sample-idx/--y-true directly).\n"
            "Generate it with e.g. MNIST/select_eval_samples.py or pass a path via --manifest."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    raise ValueError(f"Unexpected manifest format: {path}")


def manifest_sample_idx(spec: DatasetSpec, entry: dict) -> int:
    idx = entry.get("sample_idx", entry.get("dataset_index"))
    if idx is None:
        raise KeyError(f"Manifest entry missing sample_idx/dataset_index: {entry!r}")
    return int(idx)


def list_manifest_entries(
    spec: DatasetSpec,
    *,
    limit: int | None = None,
    manifest_path: Path | None = None,
) -> list[ManifestEntry]:
    path = manifest_path or spec.manifest
    entries = load_manifest(path)
    out: list[ManifestEntry] = []
    for i, raw in enumerate(entries):
        if limit is not None and i >= limit:
            break
        idx = manifest_sample_idx(spec, raw)
        if "y_true" not in raw:
            raise KeyError(f"Manifest entry missing y_true: {raw!r}")
        y_true = int(raw["y_true"])
        y_target = raw.get("y_intended_target")
        out.append(
            ManifestEntry(
                sample_idx=idx,
                y_true=y_true,
                y_intended_target=int(y_target) if y_target is not None else None,
                raw=raw,
            )
        )
    return out


def iter_manifest(
    spec: DatasetSpec,
    *,
    limit: int | None = None,
    manifest_path: Path | None = None,
) -> Iterator[ManifestEntry]:
    yield from list_manifest_entries(spec, limit=limit, manifest_path=manifest_path)
