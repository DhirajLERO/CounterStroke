"""Rewrite checkpoint_dir and ckpt_path in copied JSON configs."""

from __future__ import annotations

import json
from pathlib import Path

from paths import BENCHMARK_ROOT


def _patch_obj(obj, replacements: dict[str, str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in replacements and isinstance(v, str):
                obj[k] = replacements[k]
            else:
                _patch_obj(v, replacements)
    elif isinstance(obj, list):
        for item in obj:
            _patch_obj(item, replacements)


def patch_morphomnist() -> None:
    root = BENCHMARK_ROOT / "morphomnist" / "configs"
    scm_reps = {
        "checkpoint_dir": "./checkpoints/trained_scm",
        "ckpt_path": "./checkpoints/trained_classifiers/",
        "ckpt_cls_path": "./checkpoints/trained_classifiers/",
    }
    uncond_reps = {"checkpoint_dir": "./checkpoints/trained_uncond_vae"}
    for p in root.glob("*.json"):
        data = json.loads(p.read_text())
        _patch_obj(data, uncond_reps if "uncond" in p.stem else scm_reps)
        p.write_text(json.dumps(data, indent=4) + "\n")


def patch_celeba(graph: str) -> None:
    root = BENCHMARK_ROOT / "celeba" / "configs" / graph
    scm_reps = {
        "checkpoint_dir": f"./checkpoints/{graph}/trained_scm",
        "ckpt_path": f"./checkpoints/{graph}/trained_classifiers/",
        "ckpt_cls_path": f"./checkpoints/{graph}/trained_classifiers/",
    }
    uncond_reps = {"checkpoint_dir": f"./checkpoints/{graph}/trained_uncond_vae"}
    for p in root.glob("*.json"):
        data = json.loads(p.read_text())
        _patch_obj(data, uncond_reps if "uncond" in p.stem else scm_reps)
        p.write_text(json.dumps(data, indent=4) + "\n")


if __name__ == "__main__":
    patch_morphomnist()
    patch_celeba("simple")
    patch_celeba("complex")
    print("Patched config paths.")
