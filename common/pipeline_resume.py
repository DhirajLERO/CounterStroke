"""Phase sentinels for resumable pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sentinel_path(models_dir: Path, phase: str) -> Path:
    return models_dir / f".phase_{phase}.done"


def phase_done(models_dir: Path, phase: str) -> bool:
    return sentinel_path(models_dir, phase).is_file()


def mark_phase_done(models_dir: Path, phase: str, manifest: dict[str, Any] | None = None) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    p = sentinel_path(models_dir, phase)
    data = {"phase": phase, **(manifest or {})}
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def clear_phase(models_dir: Path, phase: str) -> None:
    p = sentinel_path(models_dir, phase)
    if p.is_file():
        p.unlink()


def hash_config(config_path: Path) -> str:
    if not config_path.is_file():
        return ""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]


def should_run_phase(
    models_dir: Path,
    phase: str,
    *,
    force: bool,
    force_phase: str | None,
    only_phase: str | None,
) -> bool:
    if only_phase and phase != only_phase:
        return False
    if force_phase:
        return phase == force_phase
    if force:
        return True
    return not phase_done(models_dir, phase)
