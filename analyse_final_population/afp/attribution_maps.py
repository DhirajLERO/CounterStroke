"""Load / compute attribution baseline importance maps for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .path_setup import ensure_explainability_on_path
from .paths import GRACOIMAGE_ROOT

EXPLAINABILITY_OUTPUTS = GRACOIMAGE_ROOT / "explainability" / "outputs"

# Default-on methods (faster). shap_gradient / occlusion opt-in via UI.
DEFAULT_ATTRIBUTION_METHODS: tuple[str, ...] = (
    "integrated_gradients",
    "input_x_gradient",
    "smoothgrad",
    "gradcam",
)

ALL_BASELINE_METHOD_IDS: tuple[str, ...] = (
    "integrated_gradients",
    "input_x_gradient",
    "smoothgrad",
    "occlusion",
    "gradcam",
    "shap_gradient",
)


@dataclass(frozen=True)
class AttributionMapResult:
    importance: np.ndarray | None
    source: str  # "precomputed" | "computed" | "error"
    error: str | None = None


def sample_dir_name(dataset: str, sample_idx: int) -> str:
    if dataset == "imagenet100":
        return f"sample_{int(sample_idx):08d}"
    return f"sample_{int(sample_idx):05d}"


def explainability_sample_dir(
    dataset: str,
    sample_idx: int,
    *,
    outputs_root: Path | None = None,
) -> Path:
    root = Path(outputs_root) if outputs_root is not None else EXPLAINABILITY_OUTPUTS
    return root / dataset / sample_dir_name(dataset, sample_idx)


def load_importance_map(sample_dir: Path, method: str) -> np.ndarray | None:
    path = sample_dir / "maps" / f"{method}.npz"
    if not path.is_file():
        return None
    try:
        z = np.load(path)
    except Exception:
        return None
    if "importance" not in z.files:
        return None
    return np.asarray(z["importance"], dtype=np.float32)


def _save_importance_map(sample_dir: Path, method: str, importance: np.ndarray) -> None:
    ensure_explainability_on_path()
    from explainability.io.artifacts import save_map
    from explainability.phenotype.diffs import importance_to_ranking

    path = sample_dir / "maps" / f"{method}.npz"
    ranking = importance_to_ranking(importance)
    save_map(path, importance, ranking)


def ensure_baseline_map(
    *,
    dataset: str,
    sample_idx: int,
    method: str,
    img_alibi: np.ndarray,
    target_class: int,
    weights: Path,
    device: torch.device | str,
    outputs_root: Path | None = None,
    cache: bool = True,
) -> AttributionMapResult:
    """Load precomputed map or compute via explainability baselines and optionally cache."""
    sample_dir = explainability_sample_dir(dataset, sample_idx, outputs_root=outputs_root)
    existing = load_importance_map(sample_dir, method)
    if existing is not None:
        return AttributionMapResult(importance=existing, source="precomputed")

    ensure_explainability_on_path()
    try:
        from explainability.baselines.methods import compute_baseline
        from explainability.config import get_dataset_spec
    except ImportError as exc:
        return AttributionMapResult(
            importance=None,
            source="error",
            error=f"explainability import failed: {exc}",
        )

    try:
        spec = get_dataset_spec(dataset)
        dev = torch.device(device) if isinstance(device, str) else device
        importance = compute_baseline(
            method,
            spec,
            img_alibi,
            int(target_class),
            Path(weights),
            dev,
        )
        importance = np.asarray(importance, dtype=np.float32)
    except Exception as exc:
        return AttributionMapResult(
            importance=None,
            source="error",
            error=f"{type(exc).__name__}: {exc}",
        )

    if cache:
        try:
            _save_importance_map(sample_dir, method, importance)
        except Exception:
            pass

    return AttributionMapResult(importance=importance, source="computed")


def resolve_baseline_maps(
    *,
    dataset: str,
    sample_idx: int,
    methods: list[str],
    img_alibi: np.ndarray,
    target_class: int,
    weights: Path,
    device: torch.device | str,
    outputs_root: Path | None = None,
    cache: bool = True,
) -> dict[str, AttributionMapResult]:
    return {
        m: ensure_baseline_map(
            dataset=dataset,
            sample_idx=sample_idx,
            method=m,
            img_alibi=img_alibi,
            target_class=target_class,
            weights=weights,
            device=device,
            outputs_root=outputs_root,
            cache=cache,
        )
        for m in methods
    }
