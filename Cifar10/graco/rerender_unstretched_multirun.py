"""Redraw pooled CIFAR CounterStroke picks on the unstretched test image."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_GRACO = Path(__file__).resolve().parent
_CIFAR = _GRACO.parent
_GRACOIMAGE = _CIFAR.parent
sys.path.insert(0, str(_CIFAR))
sys.path.insert(0, str(_GRACO))

from cifar_alibi_data import load_cifar10_alibi  # noqa: E402


@dataclass
class Row:
    o1: float
    o2: float
    o3: float
    original_class: int
    pred_class: int
    pred_prob: float
    phenotype: str


def _read_rows(csv_path: Path) -> list[Row]:
    rows: list[Row] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for rec in csv.DictReader(handle):
            rows.append(
                Row(
                    o1=float(rec["o1"]),
                    o2=float(rec["o2"]),
                    o3=float(rec["o3"]),
                    original_class=int(float(rec["original_class"])),
                    pred_class=int(float(rec["pred_class"])),
                    pred_prob=float(rec["pred_prob"]),
                    phenotype=str(rec["phenotype"]),
                )
            )
    return rows


def _pick_min_o2_then_o3(cands: list[Row]) -> Row | None:
    if not cands:
        return None
    return min(cands, key=lambda r: (r.o2, r.o3))


def _pick_min_o3_then_o2(cands: list[Row]) -> Row | None:
    if not cands:
        return None
    return min(cands, key=lambda r: (r.o3, r.o2))


def _pick_graco50(cands: list[Row]) -> Row | None:
    hi = [r for r in cands if r.pred_prob >= 0.5]
    return _pick_min_o2_then_o3(hi)


def _meta_row(row: Row | None) -> dict | None:
    if row is None:
        return None
    return {
        "pred_class": int(row.pred_class),
        "original_class": int(row.original_class),
        "pred_prob": float(row.pred_prob),
        "o1": float(row.o1),
        "o2": float(row.o2),
        "o3": float(row.o3),
    }


def _sample_dirs(graco_root: Path) -> list[tuple[int, int, Path]]:
    found: list[tuple[int, int, Path]] = []
    for path in graco_root.iterdir():
        if not path.is_dir():
            continue
        parts = path.name.split("_label_")
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
            folder_y = int(parts[1])
        except ValueError:
            continue
        found.append((idx, folder_y, path))
    found.sort()
    return found


def _render(phenotype: str, image_hwc: np.ndarray, turtle_delta_scale: float) -> np.ndarray:
    old = os.getcwd()
    os.chdir(_GRACO)
    try:
        from graco_cifar_turtle import process_phenotypes_for_images

        batch = process_phenotypes_for_images(
            [phenotype],
            image_hwc,
            clip=None,
            turtle_delta_scale=float(turtle_delta_scale),
        )
    finally:
        os.chdir(old)
    return np.asarray(batch[0], dtype=np.float32).reshape(1, 32, 32, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graco-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cifar-root", type=Path, default=_CIFAR / "data")
    parser.add_argument("--turtle-delta-scale", type=float, default=5.0)
    args = parser.parse_args()

    graco_root = args.graco_root.resolve()
    out_root = args.out_root.resolve()
    if out_root == graco_root:
        raise SystemExit("Refusing to write into --graco-root")
    sample_dirs = _sample_dirs(graco_root)
    if not sample_dirs:
        raise SystemExit(f"No {{idx}}_label_{{y}} folders in {graco_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    (_x_train, _y_train, _ohe), (x_test, y_test, _ohe2) = load_cifar10_alibi(str(args.cifar_root))
    print(f"{len(sample_dirs)} folders, indices {sample_dirs[0][0]}..{sample_dirs[-1][0]}")
    n_ok = 0
    for idx, folder_y, sample_dir in sample_dirs:
        if idx < 0 or idx >= len(x_test):
            print(f"skip {sample_dir.name}: index outside the CIFAR test set")
            continue
        y_true = int(y_test[idx])
        if y_true != folder_y:
            print(f"warning {sample_dir.name}: folder label {folder_y}, test label {y_true}")
        run_dirs = sorted(p for p in sample_dir.glob("run_*") if p.is_dir())
        rows: list[Row] = []
        for run_dir in run_dirs:
            for csv_path in sorted(run_dir.glob("*_final_generation.csv")):
                rows.extend(_read_rows(csv_path))
        flips = [r for r in rows if r.pred_class != r.original_class]
        picks = {
            "graco_base_o2": _pick_min_o2_then_o3(flips),
            "graco_base_o3": _pick_min_o3_then_o2(flips),
            "graco50": _pick_graco50(flips),
        }
        image = np.asarray(x_test[idx], dtype=np.float32)
        payload = {"orig_nhwc": image.reshape(1, 32, 32, 3)}
        for key, row in picks.items():
            if row is None:
                payload[f"cf_{key}"] = np.full((1, 32, 32, 3), np.nan, dtype=np.float32)
            else:
                payload[f"cf_{key}"] = _render(row.phenotype, image, args.turtle_delta_scale)
        out_dir = out_root / f"sample_{idx:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "tensors.npz", **payload)
        meta = {
            "sample_idx": idx,
            "y_true": y_true,
            "y_intended_target": (y_true + 1) % 10,
            "n_runs": len(run_dirs),
            "run_names": [p.name for p in run_dirs],
            "n_candidates": len(rows),
            "graco_selected_row": {key: _meta_row(row) for key, row in picks.items()},
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"sample {idx}: runs={len(run_dirs)} candidates={len(rows)} flips={len(flips)}")
        n_ok += 1
    print(f"wrote {n_ok} samples to {out_root}")


if __name__ == "__main__":
    main()