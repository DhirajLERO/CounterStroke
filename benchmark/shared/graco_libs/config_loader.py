from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None, *, defaults: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Loads a JSON config file and merges it over `defaults`."""
    cfg: Dict[str, Any] = dict(defaults or {})
    if path is None:
        return cfg

    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a JSON object, got: {type(raw).__name__}")

    return _deep_update(cfg, raw)

