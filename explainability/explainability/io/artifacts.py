"""Save and load per-sample explainability artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..paths import OUTPUTS_ROOT


def sample_output_dir(dataset: str, sample_idx: int) -> Path:
    pad = 8 if dataset == "imagenet100" else 5
    return OUTPUTS_ROOT / dataset / f"sample_{sample_idx:0{pad}d}"


def save_map(path: Path, importance: np.ndarray, ranking: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, importance=importance.astype(np.float32), ranking=ranking.astype(np.int64))


def save_srg(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        srg_score=np.array(result.srg_score, dtype=np.float32),
        mif_curve=result.mif_curve,
        lif_curve=result.lif_curve,
        k_fractions=result.k_fractions,
        n_pixels=np.array(result.n_pixels, dtype=np.int64),
        p_baseline=np.array(result.p_baseline, dtype=np.float32),
        occlusion_strategy=np.array(result.occlusion_strategy, dtype=object),
        prob_class=np.array(result.prob_class, dtype=np.int64),
    )


def save_diffs_cache(path: Path, diffs: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {f"d{i}": d.astype(np.float32) for i, d in enumerate(diffs)}
    payload["n"] = np.array(len(diffs), dtype=np.int64)
    np.savez_compressed(path, **payload)


def load_diffs_cache(path: Path) -> list[np.ndarray]:
    data = np.load(path)
    if "n" not in data:
        raise KeyError("diffs cache missing 'n'")
    n = int(data["n"])
    diffs: list[np.ndarray] = []
    for i in range(n):
        key = f"d{i}"
        if key not in data:
            raise KeyError(f"diffs cache missing {key!r}")
        diffs.append(np.asarray(data[key], dtype=np.float32))
    return diffs


def save_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def sample_artifacts_complete(
    dataset: str,
    sample_idx: int,
    methods: list[str],
    *,
    save_overlays: bool = True,
    save_srg_plots: bool = True,
) -> tuple[bool, list[str]]:
    """Return whether all expected artifacts exist for a processed sample."""
    out_dir = sample_output_dir(dataset, sample_idx)
    missing: list[str] = []

    meta_path = out_dir / "meta.json"
    if not meta_path.is_file():
        missing.append(str(meta_path.relative_to(out_dir)))

    for method in methods:
        map_path = out_dir / "maps" / f"{method}.npz"
        if not map_path.is_file():
            missing.append(str(map_path.relative_to(out_dir)))

        srg_path = out_dir / "srg" / f"{method}.npz"
        if not srg_path.is_file():
            missing.append(str(srg_path.relative_to(out_dir)))

        if save_overlays:
            overlay_path = out_dir / "overlays" / f"{method}.png"
            if not overlay_path.is_file():
                missing.append(str(overlay_path.relative_to(out_dir)))

        if save_srg_plots:
            plot_path = out_dir / "srg_plots" / f"{method}.png"
            if not plot_path.is_file():
                missing.append(str(plot_path.relative_to(out_dir)))

    if save_srg_plots and methods:
        grid_path = out_dir / "srg_plots" / "_all_methods.png"
        if not grid_path.is_file():
            missing.append(str(grid_path.relative_to(out_dir)))

    return len(missing) == 0, missing


def save_overlay(path: Path, base_hw: np.ndarray, heat: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(base_hw, cmap="gray" if base_hw.ndim == 2 else None)
    ax.imshow(heat, cmap="turbo", alpha=0.55)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
