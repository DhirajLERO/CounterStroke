"""Pool the three runs at a checkpoint and call the existing metric scripts."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from snapshots import population_csv_path, read_population_csv
from spec import (
    GRACOIMAGE_ROOT,
    RUN_IDS,
    SPECS,
    generation_root,
    load_config,
    load_manifest,
    manifest_sample_idx,
    metrics_root,
    run_dir,
)
from turtle_io import import_turtle, invalid_cf, load_image, render_phenotype, to_nhwc_alibi


@dataclass(frozen=True)
class Pick:
    o1: float
    o2: float
    o3: float
    original_class: int
    pred_class: int
    pred_prob: float
    phenotype: str


def _rows(path: Path) -> list[Pick]:
    out: list[Pick] = []
    for rec in read_population_csv(path):
        out.append(
            Pick(
                o1=float(rec["o1"]),
                o2=float(rec["o2"]),
                o3=float(rec["o3"]),
                original_class=int(float(rec["original_class"])),
                pred_class=int(float(rec["pred_class"])),
                pred_prob=float(rec["pred_prob"]),
                phenotype=str(rec["phenotype"]),
            )
        )
    return out


def _flips(rows: list[Pick]) -> list[Pick]:
    return [row for row in rows if row.pred_class != row.original_class]


def _pick_o2(rows: list[Pick]) -> Pick | None:
    if not rows:
        return None
    return min(rows, key=lambda row: (row.o2, row.o3))


def _pick_o3(rows: list[Pick]) -> Pick | None:
    if not rows:
        return None
    return min(rows, key=lambda row: (row.o3, row.o2))


def _pick_50(rows: list[Pick]) -> Pick | None:
    confident = [row for row in rows if row.pred_prob >= 0.5]
    return _pick_o2(confident)


def _missing_runs(dataset: str, entry: dict[str, Any], generation: int) -> list[int]:
    sample_idx = manifest_sample_idx(entry)
    y_true = int(entry["y_true"])
    missing = []
    for run_id in RUN_IDS:
        if not population_csv_path(run_dir(dataset, sample_idx, y_true, run_id), generation).is_file():
            missing.append(run_id)
    return missing


def _pooled_rows(dataset: str, entry: dict[str, Any], generation: int) -> list[Pick]:
    sample_idx = manifest_sample_idx(entry)
    y_true = int(entry["y_true"])
    rows: list[Pick] = []
    for run_id in RUN_IDS:
        path = population_csv_path(run_dir(dataset, sample_idx, y_true, run_id), generation)
        rows.extend(_rows(path))
    return rows


def _sample_folder(dataset: str, generation: int, sample_idx: int) -> Path:
    return generation_root(dataset, generation) / f"sample_{int(sample_idx):05d}"


def _write_sample(
    dataset: str,
    generation: int,
    entry: dict[str, Any],
    *,
    imagenet_root: str | None,
    turtle_delta_scale: float,
) -> None:
    sample_idx = manifest_sample_idx(entry)
    folder = _sample_folder(dataset, generation, sample_idx)
    npz_path = folder / "tensors.npz"
    meta_path = folder / "meta.json"
    if npz_path.is_file() and meta_path.is_file():
        newest_csv = max(
            population_csv_path(
                run_dir(dataset, sample_idx, int(entry["y_true"]), run_id),
                generation,
            ).stat().st_mtime
            for run_id in RUN_IDS
        )
        if newest_csv <= npz_path.stat().st_mtime:
            return
    turtle = import_turtle(dataset)
    idx, y_true, image, y_intended = load_image(
        dataset, turtle, entry, imagenet_root=imagenet_root
    )
    flips = _flips(_pooled_rows(dataset, entry, generation))
    picks = {
        "graco_base_o2": _pick_o2(flips),
        "graco_base_o3": _pick_o3(flips),
        "graco50": _pick_50(flips),
    }

    def pack(row: Pick | None) -> np.ndarray:
        if row is None or not row.phenotype:
            return invalid_cf(dataset)
        try:
            return render_phenotype(
                dataset,
                turtle,
                image,
                row.phenotype,
                turtle_delta_scale=turtle_delta_scale,
            )
        except Exception:
            return invalid_cf(dataset)

    payload = {
        "orig_nhwc": to_nhwc_alibi(dataset, image),
        "cf_graco_base_o2": pack(picks["graco_base_o2"]),
        "cf_graco_base_o3": pack(picks["graco_base_o3"]),
        "cf_graco50": pack(picks["graco50"]),
    }
    meta = {
        "sample_idx": int(idx),
        "y_true": int(y_true),
        "y_intended_target": int(y_intended),
        "dataset": dataset,
        "generation": int(generation),
        "run_ids": list(RUN_IDS),
        "graco_selected_row": {
            name: None
            if row is None
            else {
                "pred_class": int(row.pred_class),
                "original_class": int(row.original_class),
                "pred_prob": float(row.pred_prob),
                "o1": float(row.o1),
                "o2": float(row.o2),
                "o3": float(row.o3),
            }
            for name, row in picks.items()
        },
    }
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "tensors.npz", **payload)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(GRACOIMAGE_ROOT))


_HELP_FLAGS: dict[Path, set[str]] = {}


def _script_flags(script: Path) -> set[str]:
    """Flags this copy of the metric script accepts. Copies differ by machine."""
    cached = _HELP_FLAGS.get(script)
    if cached is not None:
        return cached
    proc = subprocess.run(
        [sys.executable, str(script), "-h"],
        cwd=str(GRACOIMAGE_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    flags = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", proc.stdout + proc.stderr))
    _HELP_FLAGS[script] = flags
    return flags


def _extend_known(cmd: list[str], script: Path, pairs: list[tuple[str, str]]) -> None:
    known = _script_flags(script)
    for flag, value in pairs:
        if flag in known:
            cmd.extend([flag, str(value)])


def _score_generation(
    dataset: str,
    generation: int,
    *,
    imagenet_root: str | None,
) -> None:
    spec = SPECS[dataset]
    gen_root = generation_root(dataset, generation)
    metrics = metrics_root(dataset, generation)
    metrics.mkdir(parents=True, exist_ok=True)
    per_sample = metrics / "per_sample_metrics.csv"
    batch_fid = metrics / "batch_fid.csv"
    distribution = metrics / "distribution_metrics.csv"
    stratified = metrics / "cmmd_batch_stratified.csv"
    tensors = list(gen_root.glob("sample_*/tensors.npz"))
    newest_tensor = max((path.stat().st_mtime for path in tensors), default=0.0)

    def _fresh(path: Path) -> bool:
        return path.is_file() and path.stat().st_mtime >= newest_tensor

    if _fresh(per_sample) and _fresh(batch_fid) and _fresh(distribution) and _fresh(stratified):
        print(f"  metrics already present for {dataset} generation {generation}", flush=True)
        return

    eval_cmd = [
        sys.executable,
        str(spec.eval_script),
        "--gen-root",
        str(gen_root),
        "--csv-out",
        str(per_sample),
        "--batch-fid-csv",
        str(batch_fid),
    ]
    cmmd_cmd = [
        sys.executable,
        str(spec.cmmd_script),
        "--gen-root",
        str(gen_root),
        "--out-dir",
        str(metrics),
    ]
    train_pool = spec.root / "data_subset" / "train_pool_index.json"
    if dataset == "mnist":
        pairs = [
            ("--pt-weights", str(spec.model_ckpt)),
            ("--mnist-root", str(spec.data_dir)),
            ("--train-pool", str(train_pool)),
        ]
        _extend_known(eval_cmd, spec.eval_script, pairs)
        _extend_known(cmmd_cmd, spec.cmmd_script, pairs)
    elif dataset == "cifar":
        pairs = [
            ("--pt-weights", str(spec.model_ckpt)),
            ("--cifar-root", str(spec.data_dir)),
        ]
        _extend_known(eval_cmd, spec.eval_script, pairs)
        _extend_known(cmmd_cmd, spec.cmmd_script, pairs)
    else:
        if not imagenet_root:
            raise SystemExit("ImageNet scoring needs --imagenet-root or IMAGENET_ROOT.")
        pairs = [
            ("--imagenet-root", imagenet_root),
            ("--classifier-weights", str(spec.model_ckpt)),
            ("--train-pool-json", str(train_pool)),
        ]
        _extend_known(eval_cmd, spec.eval_script, pairs)
        _extend_known(cmmd_cmd, spec.cmmd_script, pairs)
    if not (_fresh(per_sample) and _fresh(batch_fid)):
        _run(eval_cmd)
    if not (_fresh(distribution) and _fresh(stratified)):
        _run(cmmd_cmd)


def score_dataset(dataset: str, *, imagenet_root: str | None) -> None:
    cfg = load_config(dataset)
    generations = tuple(int(g) for g in cfg["budget"]["snapshot_gens"])
    scale = float(cfg["runtime"]["turtle_delta_scale"])
    entries = load_manifest(dataset)
    for generation in generations:
        pending = []
        blocked: list[str] = []
        for entry in entries:
            missing = _missing_runs(dataset, entry, generation)
            sample_idx = manifest_sample_idx(entry)
            if missing:
                blocked.append(f"{sample_idx} missing runs {missing}")
            else:
                pending.append(entry)
        for entry in pending:
            _write_sample(
                dataset,
                generation,
                entry,
                imagenet_root=imagenet_root,
                turtle_delta_scale=scale,
            )
        if blocked:
            print(
                f"{dataset} generation {generation}: {len(pending)} samples pooled, "
                f"{len(blocked)} still evolving. Metric scripts wait until every sample is ready.",
                flush=True,
            )
            continue
        print(f"{dataset} generation {generation}: scoring {len(entries)} samples", flush=True)
        _score_generation(dataset, generation, imagenet_root=imagenet_root)
