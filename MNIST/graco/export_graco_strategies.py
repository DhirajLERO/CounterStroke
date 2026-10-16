"""Export Graco v8 runs into sample_XXXXX/tensors.npz keys cf_graco_* (three strategies)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from torchvision.datasets import MNIST

_MNIST_DIR = Path(__file__).resolve().parents[1]
_GRACO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MNIST_DIR))
sys.path.insert(0, str(_GRACO_DIR))


def _load_process_phenotypes() -> Callable:
    old = os.getcwd()
    os.chdir(_GRACO_DIR)
    try:
        from graco_v8_turtle import process_phenotypes_for_images  # noqa: PLC0415

        return process_phenotypes_for_images
    finally:
        os.chdir(old)


@dataclass(frozen=True)
class Row:
    o1: float
    o2: float
    o3: float
    original_class: int
    pred_class: int
    pred_prob: float
    phenotype: str


def _iter_final_generation_csvs(sample_dir: Path) -> Iterator[Path]:
    if not sample_dir.is_dir():
        return
    for run_dir in sorted(sample_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        for p in sorted(run_dir.glob("*_final_generation.csv")):
            if p.is_file():
                yield p


def _read_rows(csv_path: Path) -> list[Row]:
    rows: list[Row] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for rec in r:
            rows.append(
                Row(
                    o1=float(rec["o1"]),
                    o2=float(rec["o2"]),
                    o3=float(rec["o3"]),
                    original_class=int(rec["original_class"]),
                    pred_class=int(rec["pred_class"]),
                    pred_prob=float(rec["pred_prob"]),
                    phenotype=str(rec["phenotype"]),
                )
            )
    return rows


def _flip_rows(rows: list[Row]) -> list[Row]:
    return [r for r in rows if r.pred_class != r.original_class]


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


def _row_to_selected_meta(row: Row | None) -> dict | None:
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


def _norm_mnist_image_for_sample(sample_idx: int, *, mnist_root: Path, train: bool) -> np.ndarray:
    from mnist_transforms import alibi_mnist_transform_hw  # noqa: PLC0415

    tfm = alibi_mnist_transform_hw()
    ds = MNIST(root=str(mnist_root), train=train, download=True, transform=tfm)
    img_t, _y = ds[int(sample_idx)]
    return np.ascontiguousarray(img_t.squeeze().numpy(), dtype=np.float32)


def _cf_01_to_nhwc_alibi(img_01_28: np.ndarray) -> np.ndarray:
    a = np.asarray(img_01_28, dtype=np.float32).reshape(28, 28)
    return (a - 0.5).astype(np.float32)[np.newaxis, :, :, np.newaxis]


def _invalid_cf_tensor() -> np.ndarray:
    return np.full((1, 28, 28, 1), np.nan, dtype=np.float32)


def _tensor_render_ok(cf: np.ndarray) -> bool:
    a = np.asarray(cf, dtype=np.float32)
    return bool(a.size > 0 and np.isfinite(a).any() and not np.isnan(a).all())


def _render_cf_01(
    phenotype: str,
    image_norm_hw: np.ndarray,
    process_phenotypes_for_images: Callable,
    *,
    turtle_delta_scale: float = 5.0,
) -> np.ndarray:
    from mnist_transforms import alibi_to_01  # noqa: PLC0415

    old = os.getcwd()
    os.chdir(_GRACO_DIR)
    try:
        batch = process_phenotypes_for_images(
            [phenotype],
            image_norm_hw,
            clip=None,
            turtle_delta_scale=float(turtle_delta_scale),
        )
    finally:
        os.chdir(old)
    return np.asarray(alibi_to_01(batch[0]), dtype=np.float32).reshape(28, 28)


def _find_graco_sample_dir(graco_results_root: Path, sample_idx: int, y_true: int) -> Path | None:
    exact = graco_results_root / f"{sample_idx}_label_{y_true}"
    if exact.is_dir():
        return exact
    matches = sorted(graco_results_root.glob(f"{sample_idx}_label_*"))
    for m in matches:
        suffix = m.name.split("_label_", 1)[-1]
        try:
            y = int(suffix)
        except ValueError:
            continue
        if y == y_true:
            return m
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-gen-root", type=Path, required=True)
    p.add_argument("--graco-results-root", type=Path, required=True)
    p.add_argument("--out-gen-root", type=Path, required=True)
    p.add_argument("--mnist-root", type=Path, default=_MNIST_DIR / "data")
    p.add_argument("--mnist-train", action="store_true")
    p.add_argument("--turtle-delta-scale", type=float, default=5.0)
    args = p.parse_args()

    process_phenotypes_for_images = _load_process_phenotypes()
    base = args.base_gen_root.resolve()
    graco_root = args.graco_results_root.resolve()
    out_root = args.out_gen_root.resolve()
    train_split = bool(args.mnist_train)

    if not base.is_dir():
        raise SystemExit(f"Missing base {base}")
    if not graco_root.is_dir():
        raise SystemExit(f"Missing graco {graco_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for sdir in sorted(base.glob("sample_*")):
        meta_path = sdir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample_idx = int(meta["sample_idx"])
        y_true = int(meta["y_true"])

        graco_sample_dir = _find_graco_sample_dir(graco_root, sample_idx, y_true)
        all_rows: list[Row] = []
        if graco_sample_dir is not None:
            for csv_path in _iter_final_generation_csvs(graco_sample_dir):
                try:
                    all_rows.extend(_read_rows(csv_path))
                except (KeyError, ValueError):
                    pass

        flips = _flip_rows(all_rows)
        r_o2 = _pick_min_o2_then_o3(flips)
        r_o3 = _pick_min_o3_then_o2(flips)
        r_50 = _pick_graco50(flips)

        image_norm = _norm_mnist_image_for_sample(
            sample_idx, mnist_root=args.mnist_root, train=train_split
        )

        def pack(row: Row | None) -> np.ndarray:
            if row is None:
                return _invalid_cf_tensor()
            try:
                img01 = _render_cf_01(
                    row.phenotype,
                    image_norm,
                    process_phenotypes_for_images,
                    turtle_delta_scale=args.turtle_delta_scale,
                )
                return _cf_01_to_nhwc_alibi(img01)
            except Exception:
                return _invalid_cf_tensor()

        cf_o2 = pack(r_o2)
        cf_o3 = pack(r_o3)
        cf50 = pack(r_50)

        meta["graco_strategy_valid"] = {
            "graco_base_o2": int(len(flips) > 0),
            "graco_base_o3": int(len(flips) > 0),
            "graco50": int(any(r.pred_prob >= 0.5 for r in flips)),
        }
        meta["graco_render_ok"] = {
            "graco_base_o2": int(_tensor_render_ok(cf_o2)),
            "graco_base_o3": int(_tensor_render_ok(cf_o3)),
            "graco50": int(_tensor_render_ok(cf50)),
        }
        meta["graco_selected_row"] = {
            "graco_base_o2": _row_to_selected_meta(r_o2),
            "graco_base_o3": _row_to_selected_meta(r_o3),
            "graco50": _row_to_selected_meta(r_50),
        }

        base_npz = sdir / "tensors.npz"
        payload: dict[str, np.ndarray] = {}
        if base_npz.is_file():
            z = np.load(base_npz)
            for k in z.files:
                payload[k] = np.asarray(z[k])
            orig = np.asarray(z["orig_nhwc"], dtype=np.float32)
        else:
            ds0 = MNIST(root=str(args.mnist_root), train=train_split, download=True, transform=None)
            pix = ds0[sample_idx][0]
            arr = (np.asarray(pix, dtype=np.float32) / 255.0).reshape(28, 28, 1) - 0.5
            orig = arr[np.newaxis, ...].astype(np.float32)
            payload["orig_nhwc"] = orig

        payload["orig_nhwc"] = orig
        payload["cf_graco_base_o2"] = cf_o2
        payload["cf_graco_base_o3"] = cf_o3
        payload["cf_graco50"] = cf50

        out_sdir = out_root / sdir.name
        out_sdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_sdir / "tensors.npz", **payload)
        (out_sdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        n_ok += 1

    print(f"Exported {n_ok} samples to {out_root}")


if __name__ == "__main__":
    main()
