"""Dataset registry for factual Graco runs and blank-canvas analysis outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .paths import ANALYSIS_ROOT, CIFAR_DIR, GRACOIMAGE_ROOT, IMNET_DIR, MNIST_DIR


def _experiment_name(config_path: Path, key: str) -> str:
    if not config_path.is_file():
        return ""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    block = cfg.get(key, {})
    return str(block.get("experiment_name", ""))


@dataclass(frozen=True)
class AppDatasetSpec:
    name: str
    num_classes: int
    graco_dir: Path
    dataset_dir: Path
    default_data_root: Path
    default_weights: Path
    default_manifest: Path
    default_factual_graco_root: Path
    default_blank_graco_root: Path
    default_gen_root: Path
    default_results_dir: Path
    experiment_config_key: str
    turtle_delta_scale: float


def get_app_dataset_spec(name: str) -> AppDatasetSpec:
    if name == "mnist":
        cfg = MNIST_DIR / "config.json"
        exp = _experiment_name(cfg, "__parallel_jobs_v8__") or "experiment_v8_mnist"
        runtime = json.loads(cfg.read_text(encoding="utf-8")).get("runtime", {}) if cfg.is_file() else {}
        return AppDatasetSpec(
            name="mnist",
            num_classes=10,
            graco_dir=MNIST_DIR / "graco",
            dataset_dir=MNIST_DIR,
            default_data_root=MNIST_DIR / "data",
            default_weights=MNIST_DIR / "models" / "mnist_cnn.pt",
            default_manifest=MNIST_DIR / "data_subset" / "eval_manifest.json",
            default_factual_graco_root=MNIST_DIR / "graco" / "results" / exp,
            default_blank_graco_root=ANALYSIS_ROOT / "mnist" / "outputs" / "graco_run",
            default_gen_root=MNIST_DIR / "outputs" / "generation",
            default_results_dir=MNIST_DIR / "outputs" / "results" / "generation",
            experiment_config_key="__parallel_jobs_v8__",
            turtle_delta_scale=float(runtime.get("turtle_delta_scale", 5.0)),
        )
    if name == "cifar10":
        cfg = CIFAR_DIR / "config.json"
        exp = _experiment_name(cfg, "__parallel_jobs_v8__") or "experiment_cifar_turtle"
        runtime = json.loads(cfg.read_text(encoding="utf-8")).get("runtime", {}) if cfg.is_file() else {}
        return AppDatasetSpec(
            name="cifar10",
            num_classes=10,
            graco_dir=CIFAR_DIR / "graco",
            dataset_dir=CIFAR_DIR,
            default_data_root=CIFAR_DIR / "data",
            default_weights=CIFAR_DIR / "models" / "cifar_resnet.pt",
            default_manifest=CIFAR_DIR / "data_subset" / "eval_manifest.json",
            default_factual_graco_root=CIFAR_DIR / "graco" / "results" / exp,
            default_blank_graco_root=ANALYSIS_ROOT / "cifar10" / "outputs" / "graco_run",
            default_gen_root=CIFAR_DIR / "outputs" / "generation",
            default_results_dir=CIFAR_DIR / "outputs" / "results" / "generation",
            experiment_config_key="__parallel_jobs_v8__",
            turtle_delta_scale=float(runtime.get("turtle_delta_scale", 5.0)),
        )
    if name == "imagenet100":
        cfg = IMNET_DIR / "config.json"
        exp = _experiment_name(cfg, "__parallel_jobs_inception__") or "experiment_inception_turtle"
        runtime = json.loads(cfg.read_text(encoding="utf-8")).get("runtime", {}) if cfg.is_file() else {}
        return AppDatasetSpec(
            name="imagenet100",
            num_classes=100,
            graco_dir=IMNET_DIR / "graco",
            dataset_dir=IMNET_DIR,
            default_data_root=IMNET_DIR / "data" / "imagenet100",
            default_weights=IMNET_DIR / "models" / "imagenet100_inception.pt",
            default_manifest=IMNET_DIR / "data_subset" / "eval_manifest.json",
            default_factual_graco_root=IMNET_DIR / "graco" / "results" / exp,
            default_blank_graco_root=ANALYSIS_ROOT / "imagenet100" / "outputs" / "graco_run",
            default_gen_root=IMNET_DIR / "outputs" / "generation",
            default_results_dir=IMNET_DIR / "outputs" / "results" / "generation",
            experiment_config_key="__parallel_jobs_inception__",
            turtle_delta_scale=float(runtime.get("turtle_delta_scale", 5.0)),
        )
    raise ValueError(f"Unknown dataset {name!r}")


def resolve_factual_graco_root(spec: AppDatasetSpec, override: Path | None) -> Path:
    if override is not None:
        p = override.expanduser().resolve()
        if p.is_dir():
            return p
    for candidate in (
        spec.default_factual_graco_root,
        spec.dataset_dir / "results" / spec.default_factual_graco_root.name,
    ):
        if candidate.is_dir():
            return candidate.resolve()
    return spec.default_factual_graco_root.resolve()


def resolve_blank_graco_root(spec: AppDatasetSpec, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return spec.default_blank_graco_root.resolve()
