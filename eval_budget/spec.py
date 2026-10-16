"""Paths and dataset records for the evaluation-budget experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
GRACOIMAGE_ROOT = EVAL_ROOT.parent
MPLCONFIG = EVAL_ROOT / ".mplconfig"

RUN_IDS = (0, 1, 2)
SNAPSHOT_GENS = (100, 200, 300)
GRACO_METHODS = ("graco_base_o2", "graco_base_o3", "graco50")
METHOD_LABELS = {
    "graco_base_o2": "CS_sparse",
    "graco_base_o3": "CS_prox",
    "graco50": "CS_50",
}
PENALTY_CUTOFF = 1e5
OBJECTIVE_COLUMNS = ("avg_o1", "avg_o2", "avg_o3", "min_o1", "min_o2", "min_o3")

PER_SAMPLE_COLUMNS = (
    "has_valid_cf",
    "l2_distance",
    "sparsity_frac",
    "ssim_to_original",
    "lpips_to_original",
    "p_pred_cf",
    "ssim_to_random_pred_class",
    "lpips_to_random_pred_class",
    "inception_sqdist_to_source_class_mean",
    "inception_sqdist_to_predicted_class_mean",
)

LOWER_IS_BETTER = {
    "l2_distance",
    "sparsity_count",
    "sparsity_frac",
    "lpips_to_original",
    "lpips_to_random_pred_class",
    "inception_sqdist_to_source_class_mean",
    "inception_sqdist_to_predicted_class_mean",
    "delta_p_true",
    "p_true_cf",
    "frechet_inception_distance_vs_source_y_true",
    "frechet_inception_distance_vs_predicted_class",
    "frechet_inception_distance_cf_vs_originals",
    "kid_mean_cf_vs_originals",
    "kid_std_cf_vs_originals",
    "cmmd_orig_vs_cf_distribution",
    "cmmd_vs_stratified_source_y_true",
    "cmmd_vs_stratified_predicted_class",
    "cmmd_vs_source_y_true_class",
    "cmmd_vs_predicted_class",
}
HIGHER_IS_BETTER = {
    "has_valid_cf",
    "validity",
    "ssim_to_original",
    "ssim_to_random_pred_class",
    "p_pred_cf",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    json_key: str
    root: Path
    graco_dir: Path
    turtle_py: Path
    config_path: Path
    manifest: Path
    model_ckpt: Path
    data_dir: Path
    eval_script: Path
    cmmd_script: Path
    spatial_shape: tuple[int, ...]


SPECS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        key="mnist",
        json_key="mnist",
        root=GRACOIMAGE_ROOT / "MNIST",
        graco_dir=GRACOIMAGE_ROOT / "MNIST" / "graco",
        turtle_py=GRACOIMAGE_ROOT / "MNIST" / "graco" / "graco_v8_turtle.py",
        config_path=EVAL_ROOT / "configs" / "mnist.json",
        manifest=GRACOIMAGE_ROOT / "MNIST" / "data_subset" / "eval_manifest.json",
        model_ckpt=GRACOIMAGE_ROOT / "MNIST" / "models" / "mnist_cnn.pt",
        data_dir=GRACOIMAGE_ROOT / "MNIST" / "data",
        eval_script=GRACOIMAGE_ROOT / "common" / "eval_per_sample_mnist.py",
        cmmd_script=GRACOIMAGE_ROOT / "common" / "run_cmmd_mnist.py",
        spatial_shape=(28, 28, 1),
    ),
    "cifar": DatasetSpec(
        key="cifar",
        json_key="cifar10",
        root=GRACOIMAGE_ROOT / "Cifar10",
        graco_dir=GRACOIMAGE_ROOT / "Cifar10" / "graco",
        turtle_py=GRACOIMAGE_ROOT / "Cifar10" / "graco" / "graco_cifar_turtle.py",
        config_path=EVAL_ROOT / "configs" / "cifar.json",
        manifest=GRACOIMAGE_ROOT / "Cifar10" / "data_subset" / "eval_manifest.json",
        model_ckpt=GRACOIMAGE_ROOT / "Cifar10" / "models" / "cifar_resnet.pt",
        data_dir=GRACOIMAGE_ROOT / "Cifar10" / "data",
        eval_script=GRACOIMAGE_ROOT / "common" / "eval_per_sample_cifar10.py",
        cmmd_script=GRACOIMAGE_ROOT / "common" / "run_cmmd_cifar10.py",
        spatial_shape=(32, 32, 3),
    ),
    "imagenet": DatasetSpec(
        key="imagenet",
        json_key="imagenet100",
        root=GRACOIMAGE_ROOT / "Imagenet",
        graco_dir=GRACOIMAGE_ROOT / "Imagenet" / "graco",
        turtle_py=GRACOIMAGE_ROOT / "Imagenet" / "graco" / "graco_inception_turtle.py",
        config_path=EVAL_ROOT / "configs" / "imagenet.json",
        manifest=GRACOIMAGE_ROOT / "Imagenet" / "data_subset" / "eval_manifest.json",
        model_ckpt=GRACOIMAGE_ROOT / "Imagenet" / "models" / "imagenet100_inception.pt",
        data_dir=GRACOIMAGE_ROOT / "Imagenet" / "data" / "imagenet100",
        eval_script=GRACOIMAGE_ROOT / "common" / "eval_per_sample_imagenet.py",
        cmmd_script=GRACOIMAGE_ROOT / "common" / "run_cmmd_imagenet.py",
        spatial_shape=(299, 299, 3),
    ),
}


def load_config(dataset: str) -> dict[str, Any]:
    spec = SPECS[dataset]
    return json.loads(spec.config_path.read_text(encoding="utf-8"))


def runs_root(dataset: str) -> Path:
    return EVAL_ROOT / "runs" / dataset


def sample_dir(dataset: str, sample_idx: int, y_true: int) -> Path:
    return runs_root(dataset) / f"{int(sample_idx)}_label_{int(y_true)}"


def run_dir(dataset: str, sample_idx: int, y_true: int, run_id: int) -> Path:
    return sample_dir(dataset, sample_idx, y_true) / f"run_{int(run_id)}"


def generation_root(dataset: str, generation: int) -> Path:
    return EVAL_ROOT / "generation" / dataset / f"g{int(generation)}"


def metrics_root(dataset: str, generation: int) -> Path:
    return EVAL_ROOT / "metrics" / dataset / f"g{int(generation)}"


def comparison_root(dataset: str) -> Path:
    return EVAL_ROOT / "comparison" / dataset


def plots_root(dataset: str) -> Path:
    return EVAL_ROOT / "plots" / dataset


def evolution_seed(run_id: int) -> int:
    """Same mapping as ``common.objective_generation_log.evolution_seed``."""
    return 10 + int(run_id)


def manifest_sample_idx(entry: dict[str, Any]) -> int:
    if "dataset_index" in entry and entry["dataset_index"] is not None:
        return int(entry["dataset_index"])
    return int(entry["sample_idx"])


def load_manifest(dataset: str) -> list[dict[str, Any]]:
    spec = SPECS[dataset]
    if not spec.manifest.is_file():
        raise FileNotFoundError(
            f"Missing eval manifest {spec.manifest}. "
            "The budget run uses the same 100 images as the main CounterStroke run."
        )
    payload = json.loads(spec.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a JSON list: {spec.manifest}")
    return payload
