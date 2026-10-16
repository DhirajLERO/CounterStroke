"""Resolve unconditional VAE checkpoint paths for benchmark evaluation."""

from __future__ import annotations

import os
from pathlib import Path

ENV_UNCOND_VAE_CKPT = "BENCHMARK_UNCOND_VAE_CKPT"
ENV_CELEBA_GRAPH = "BENCHMARK_CELEBA_GRAPH"

# Upstream default filenames (Google Drive release)
DEFAULT_CKPT_NAMES = {
    "morphomnist": "uncond_image_vae-epoch=269.ckpt",
    ("celeba", "simple"): "image_vae-epoch=44.ckpt",
    ("celeba", "complex"): "image_vae-epoch=44.ckpt",
}


def default_uncond_vae_dir(dataset_root: Path, graph: str | None = None) -> Path:
    if graph:
        return dataset_root / "checkpoints" / graph / "trained_uncond_vae"
    return dataset_root / "checkpoints" / "trained_uncond_vae"


def _pick_ckpt_in_dir(directory: Path, preferred_name: str | None) -> Path:
    if preferred_name and (directory / preferred_name).is_file():
        return directory / preferred_name
    candidates = sorted(directory.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files in {directory}")
    return candidates[-1]


def resolve_uncond_vae_ckpt(
    user_path: str | Path | None,
    dataset_root: Path,
    *,
    dataset: str,
    graph: str | None = None,
) -> Path | None:
    """Resolve checkpoint file from CLI path, dir, or default location under dataset_root."""
    if user_path is not None and str(user_path).strip() == "":
        user_path = None

    if user_path is not None:
        p = Path(user_path)
        if not p.is_absolute():
            p = (dataset_root / p).resolve()
        else:
            p = p.resolve()
        if p.is_dir():
            key = (dataset, graph) if dataset == "celeba" else dataset
            pref = DEFAULT_CKPT_NAMES.get(key) if isinstance(key, tuple) else DEFAULT_CKPT_NAMES.get(key)
            return _pick_ckpt_in_dir(p, pref)
        if not p.is_file():
            raise FileNotFoundError(f"Unconditional VAE checkpoint not found: {p}")
        return p

    d = default_uncond_vae_dir(dataset_root, graph)
    if not d.is_dir():
        return None
    key: str | tuple[str, str] = (dataset, graph) if dataset == "celeba" and graph else dataset
    pref = DEFAULT_CKPT_NAMES.get(key)
    return _pick_ckpt_in_dir(d, pref)


def apply_uncond_vae_env(
    env: dict,
    *,
    embeddings: str | None,
    uncond_vae_ckpt: Path | None,
    celeba_graph: str | None = None,
) -> dict:
    """Set environment variables consumed by evaluation/embeddings/embeddings.py."""
    if embeddings:
        env["BENCHMARK_EMBEDDINGS"] = embeddings
    if embeddings == "vae" and uncond_vae_ckpt is not None:
        env[ENV_UNCOND_VAE_CKPT] = str(uncond_vae_ckpt.resolve())
    if celeba_graph:
        env[ENV_CELEBA_GRAPH] = celeba_graph
    return env


def load_uncond_vae_env_from_os() -> tuple[str | None, Path | None, str | None]:
    emb = os.environ.get("BENCHMARK_EMBEDDINGS")
    ckpt = os.environ.get(ENV_UNCOND_VAE_CKPT)
    graph = os.environ.get(ENV_CELEBA_GRAPH)
    return emb, Path(ckpt) if ckpt else None, graph
