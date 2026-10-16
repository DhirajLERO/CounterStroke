"""Load dataset configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import CONFIG_PATH, PACKAGE_ROOT

_BUILTIN_DEFAULTS = {
    "prob_threshold": 0.5,
    "cochange_tau": 0.5,
    "spatial_sigma": 1.5,
    "max_render": 200,
    "eps": 1.0e-6,
    "srg_curve_points": 100,
    "srg_infer_batch_size": 32,
    "srg_occlusion": "mean_alibi",
    "srg_occlusion_noise_std": 0.15,
    "srg_graco_prob_class": "base",
    "pagerank_backend": "implicit",
    "pagerank_damping": 0.85,
}

_DATASET_META: dict[str, dict[str, Any]] = {
    "mnist": {
        "num_classes": 10,
        "dataset_dir": "MNIST",
        "graco_dir": "MNIST/graco",
        "weights": "MNIST/models/mnist_cnn.pt",
        "manifest": "MNIST/data_subset/eval_manifest.json",
        "graco_results": "MNIST/graco/results/experiment_v8_mnist",
        "json_exp_key": "__parallel_jobs_v8__",
        "default_experiment": "experiment_v8_mnist",
        "sample_idx_key": "sample_idx",
    },
    "cifar10": {
        "num_classes": 10,
        "dataset_dir": "Cifar10",
        "graco_dir": "Cifar10/graco",
        "weights": "Cifar10/models/cifar_resnet.pt",
        "manifest": "Cifar10/data_subset/eval_manifest.json",
        "graco_results": "Cifar10/graco/results/experiment_cifar_turtle",
        "json_exp_key": "__parallel_jobs_v8__",
        "default_experiment": "experiment_cifar_turtle",
        "sample_idx_key": "sample_idx",
    },
    "imagenet100": {
        "num_classes": 100,
        "dataset_dir": "Imagenet",
        "graco_dir": "Imagenet/graco",
        "weights": "Imagenet/models/imagenet100_inception.pt",
        "manifest": "Imagenet/data_subset/eval_manifest.json",
        "graco_results": "Imagenet/graco/results/experiment_inception_turtle",
        "json_exp_key": "__parallel_jobs_inception__",
        "default_experiment": "experiment_inception_turtle",
        "sample_idx_key": "dataset_index",
    },
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    graco_dir: Path
    dataset_dir: Path
    weights: Path
    manifest: Path
    graco_results: Path
    turtle_delta_scale: float
    sample_idx_key: str


def _load_yaml_config() -> dict:
    if not CONFIG_PATH.is_file():
        return {}
    text = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError:
        return {}


def gracoimage_root(cfg: dict | None = None) -> Path:
    """GracoImage repo root; ``gracoimage_root`` in YAML is relative to this package."""
    cfg = cfg if cfg is not None else _load_yaml_config()
    rel = str(cfg.get("gracoimage_root", ".."))
    return (PACKAGE_ROOT / rel).resolve()


def resolve_config_path(value: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a YAML path (absolute, or relative to ``gracoimage_root``)."""
    root = root or gracoimage_root()
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def _experiment_name(config_path: Path, key: str) -> str:
    if not config_path.is_file():
        return ""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return str(cfg.get(key, {}).get("experiment_name", ""))


def _runtime_from_json(dataset_dir: Path, json_exp_key: str) -> dict:
    cfg_path = dataset_dir / "config.json"
    if not cfg_path.is_file():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8")).get("runtime", {}) or {}


def _graco_results_path(
    meta: dict[str, Any],
    ds_block: dict[str, Any],
    *,
    root: Path,
    dataset_dir: Path,
) -> Path:
    if ds_block.get("graco_results"):
        return resolve_config_path(ds_block["graco_results"], root=root)

    exp = _experiment_name(dataset_dir / "config.json", meta["json_exp_key"]) or str(
        meta["default_experiment"]
    )
    rel = Path(meta["dataset_dir"]) / "graco" / "results" / exp
    return resolve_config_path(rel, root=root)


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in _DATASET_META:
        raise ValueError(f"Unknown dataset {name!r}")

    yaml_cfg = _load_yaml_config()
    root = gracoimage_root(yaml_cfg)
    meta = _DATASET_META[name]
    ds_block = (yaml_cfg.get("datasets") or {}).get(name, {}) or {}

    dataset_dir = resolve_config_path(
        ds_block.get("dataset_dir", meta["dataset_dir"]),
        root=root,
    )
    graco_dir = resolve_config_path(
        ds_block.get("graco_dir", meta["graco_dir"]),
        root=root,
    )
    weights = resolve_config_path(
        ds_block.get("weights", meta["weights"]),
        root=root,
    )
    manifest = resolve_config_path(
        ds_block.get("manifest", meta["manifest"]),
        root=root,
    )
    graco_results = _graco_results_path(
        meta,
        ds_block,
        root=root,
        dataset_dir=dataset_dir,
    )

    runtime = _runtime_from_json(dataset_dir, meta["json_exp_key"])
    turtle_delta_scale = float(
        ds_block.get(
            "turtle_delta_scale",
            runtime.get("turtle_delta_scale", 5.0),
        )
    )

    return DatasetSpec(
        name=name,
        num_classes=int(ds_block.get("num_classes", meta["num_classes"])),
        graco_dir=graco_dir,
        dataset_dir=dataset_dir,
        weights=weights,
        manifest=manifest,
        graco_results=graco_results,
        turtle_delta_scale=turtle_delta_scale,
        sample_idx_key=str(ds_block.get("sample_idx_key", meta["sample_idx_key"])),
    )


def get_defaults() -> dict:
    merged = dict(_BUILTIN_DEFAULTS)
    yaml_defaults = _load_yaml_config().get("defaults") or {}
    merged.update(yaml_defaults)
    # Legacy key
    if "occlusion" in yaml_defaults and "srg_occlusion" not in yaml_defaults:
        merged["srg_occlusion"] = yaml_defaults["occlusion"]
    return merged


def resolve_graco_results(
    spec: DatasetSpec,
    override: Path | None = None,
    *,
    sample_idx: int | None = None,
    y_true: int | None = None,
) -> Path:
    """Resolve evolution results root, optionally picking the root that contains a sample."""
    from .data.graco_paths import find_graco_sample_dir

    if override is not None:
        p = override.expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"--graco-results not found: {p}")
        return p

    candidates = candidate_graco_roots(spec)

    if sample_idx is not None and y_true is not None:
        for cand in candidates:
            if cand.is_dir() and find_graco_sample_dir(cand, sample_idx, y_true) is not None:
                return cand.resolve()

    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()

    return candidates[0].resolve() if candidates else spec.graco_results.resolve()


def candidate_graco_roots(spec: DatasetSpec) -> list[Path]:
    """Search locations where factual Graco evolution outputs may live."""
    seen: set[str] = set()
    roots: list[Path] = []

    def add(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    add(spec.graco_results)
    add(spec.graco_dir / "results" / spec.graco_results.name)
    add(spec.dataset_dir / "graco" / "results" / spec.graco_results.name)
    add(spec.dataset_dir / "results" / spec.graco_results.name)
    add(spec.dataset_dir / "outputs" / "graco_v8")
    add(spec.dataset_dir / "outputs" / "graco_results")

    for base in (spec.graco_dir / "results", spec.dataset_dir / "outputs"):
        if base.is_dir():
            for child in sorted(base.iterdir()):
                if child.is_dir():
                    add(child)

    return roots
