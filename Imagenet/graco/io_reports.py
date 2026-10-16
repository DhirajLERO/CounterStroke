"""CSV / NPZ export helpers (copied from GracoImage v8 pattern, RGB-safe)."""

from __future__ import annotations

import csv

import numpy as np


def min_index_where(mask: np.ndarray, values: np.ndarray) -> int | None:
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return None
    return int(idxs[np.argmin(values[idxs])])


def write_final_generation_csv(
    *,
    out_csv_path: str,
    phenotypes: list[str],
    o1: np.ndarray,
    o2: np.ndarray,
    o3: np.ndarray,
    original_class: int,
    p_original: np.ndarray,
    pred_class: np.ndarray,
    pred_prob: np.ndarray,
) -> None:
    header = [
        "ind_index",
        "phenotype",
        "o1",
        "o2",
        "o3",
        "original_class",
        "p_original_class",
        "pred_class",
        "pred_prob",
    ]
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, ph in enumerate(phenotypes):
            w.writerow(
                [
                    i,
                    ph,
                    float(o1[i]),
                    float(o2[i]),
                    float(o3[i]),
                    int(original_class),
                    float(p_original[i]),
                    int(pred_class[i]),
                    float(pred_prob[i]),
                ]
            )


def write_run_summary_npz(
    *,
    out_npz_path: str,
    device: str,
    original_class: int,
    original_image_01: np.ndarray,
    phenotypes: list[str],
    imgs_01: np.ndarray,
    o1: np.ndarray,
    o2: np.ndarray,
    o3: np.ndarray,
    pred_class: np.ndarray,
    pred_prob: np.ndarray,
    p_original: np.ndarray,
) -> None:
    idx_best_o1 = int(np.argmin(o1))
    idx_best_o2 = int(np.argmin(o2))
    idx_best_o3 = int(np.argmin(o3))

    mask_cf = pred_class != int(original_class)
    mask_cf_p05 = mask_cf & (pred_prob > 0.5)

    idx_cf_min_o2 = min_index_where(mask_cf, o2)
    idx_cf_min_o3 = min_index_where(mask_cf, o3)
    idx_cf_p05_min_o2 = min_index_where(mask_cf_p05, o2)
    idx_cf_p05_min_o3 = min_index_where(mask_cf_p05, o3)

    def pack(idx: int | None, prefix: str) -> dict:
        if idx is None:
            return {
                f"has_{prefix}": np.array(False),
                f"{prefix}_image": np.array([], dtype=np.float32),
                f"{prefix}_phenotype": np.array("", dtype=object),
                f"{prefix}_o1": np.array(np.nan, dtype=np.float32),
                f"{prefix}_o2": np.array(np.nan, dtype=np.float32),
                f"{prefix}_o3": np.array(np.nan, dtype=np.float32),
                f"{prefix}_pred_class": np.array(-1, dtype=np.int64),
                f"{prefix}_pred_prob": np.array(np.nan, dtype=np.float32),
                f"{prefix}_p_original": np.array(np.nan, dtype=np.float32),
            }
        return {
            f"has_{prefix}": np.array(True),
            f"{prefix}_image": imgs_01[idx].astype(np.float32, copy=False),
            f"{prefix}_phenotype": np.array(phenotypes[idx], dtype=object),
            f"{prefix}_o1": np.array(float(o1[idx]), dtype=np.float32),
            f"{prefix}_o2": np.array(float(o2[idx]), dtype=np.float32),
            f"{prefix}_o3": np.array(float(o3[idx]), dtype=np.float32),
            f"{prefix}_pred_class": np.array(int(pred_class[idx]), dtype=np.int64),
            f"{prefix}_pred_prob": np.array(float(pred_prob[idx]), dtype=np.float32),
            f"{prefix}_p_original": np.array(float(p_original[idx]), dtype=np.float32),
        }

    payload: dict = {
        "device": np.array(str(device), dtype=object),
        "original_class": np.array(int(original_class), dtype=np.int64),
        "original_image": original_image_01.astype(np.float32, copy=False),
    }
    payload.update(pack(idx_best_o1, "best_o1"))
    payload.update(pack(idx_best_o2, "best_o2"))
    payload.update(pack(idx_best_o3, "best_o3"))
    payload.update(pack(idx_cf_min_o2, "cf_min_o2"))
    payload.update(pack(idx_cf_min_o3, "cf_min_o3"))
    payload.update(pack(idx_cf_p05_min_o2, "cf_p05_min_o2"))
    payload.update(pack(idx_cf_p05_min_o3, "cf_p05_min_o3"))
    np.savez_compressed(out_npz_path, **payload)
