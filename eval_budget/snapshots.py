"""Per-generation objective log, population snapshots, and resume checkpoints."""

from __future__ import annotations

import csv
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

from spec import OBJECTIVE_COLUMNS, PENALTY_CUTOFF

POPULATION_COLUMNS = (
    "ind_index",
    "phenotype",
    "o1",
    "o2",
    "o3",
    "original_class",
    "p_original_class",
    "pred_class",
    "pred_prob",
)


def _triple(row: Any) -> list[float]:
    arr = np.asarray(row, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - int(arr.size)), constant_values=np.nan)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def write_objectives(run_dir: Path, logbook: Any) -> None:
    """Rewrite the objective CSV from the logbook so a resume cannot duplicate rows."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "objectives_by_generation.csv"
    gens = list(logbook.select("gen"))
    avgs = list(logbook.select("avg_obj"))
    mins = list(logbook.select("min_obj"))
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gen", *OBJECTIVE_COLUMNS])
        for i, gen in enumerate(gens):
            writer.writerow([int(gen), *_triple(avgs[i]), *_triple(mins[i])])
    os.replace(tmp, path)


def capture_rng() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    import torch

    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.pkl"


def save_checkpoint(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(run_dir)
    tmp = run_dir / "checkpoint.pkl.tmp"
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    path = checkpoint_path(run_dir)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "completed_gen" not in payload:
        raise ValueError(f"Unreadable checkpoint {path}")
    return payload


def population_csv_path(run_dir: Path, generation: int) -> Path:
    return run_dir / f"gen_{int(generation)}_population.csv"


def write_population_csv(
    path: Path,
    *,
    phenotypes: list[str],
    o1: np.ndarray,
    o2: np.ndarray,
    o3: np.ndarray,
    original_class: int,
    p_original: np.ndarray,
    pred_class: np.ndarray,
    pred_prob: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(POPULATION_COLUMNS)
        for i, phenotype in enumerate(phenotypes):
            writer.writerow(
                [
                    i,
                    phenotype,
                    float(o1[i]),
                    float(o2[i]),
                    float(o3[i]),
                    int(original_class),
                    float(p_original[i]),
                    int(pred_class[i]),
                    float(pred_prob[i]),
                ]
            )
    os.replace(tmp, path)


def read_population_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def done_path(run_dir: Path) -> Path:
    return run_dir / "done.json"


def is_done(run_dir: Path, *, max_generations: int, snapshot_gens: tuple[int, ...]) -> bool:
    path = done_path(run_dir)
    if not path.is_file():
        return False
    try:
        payload = json_load(path)
    except (OSError, ValueError):
        return False
    if int(payload.get("max_generations", -1)) != int(max_generations):
        return False
    return all(population_csv_path(run_dir, gen).is_file() for gen in snapshot_gens)


def json_load(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def clear_partial(run_dir: Path) -> None:
    """Drop logs from a checkpoint that does not belong to this job."""
    for path in run_dir.glob("gen_*_population.csv"):
        path.unlink()
    for name in ("objectives_by_generation.csv", "done.json"):
        path = run_dir / name
        if path.is_file():
            path.unlink()


def write_done(run_dir: Path, payload: dict[str, Any]) -> None:
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    path = done_path(run_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def mask_penalty(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out[~np.isfinite(out) | (out >= PENALTY_CUTOFF)] = np.nan
    return out
