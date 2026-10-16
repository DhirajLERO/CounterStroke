"""Per-sample metrics for ImageNet RGB tensors.npz under sample_* folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_GRACOIMAGE_ROOT = Path(__file__).resolve().parents[1]
_IMNET_PKG = _GRACOIMAGE_ROOT / "Imagenet"
sys.path.insert(0, str(_GRACOIMAGE_ROOT))
sys.path.insert(0, str(_IMNET_PKG))

from common.inception_features import (  # noqa: E402
    build_inception_feature_net,
    inception_features_from_rgb_nhwc,
)
from explainers.imagenet100_inception import load_imagenet100_classifier  # noqa: E402
from imagenet_alibi_config import CKPT_IMAGENET100_INCEPTION  # noqa: E402
from imagenet100_dataset import ImageNet100AlibiDataset, load_train_pool_indices  # noqa: E402
from imagenet_data import add_imagenet_root_args, resolve_class_map_path  # noqa: E402

METHOD_NPZ_KEYS = [
    ("counterfactual_pt", "cf_counterfactual_pt"),
    ("cfproto", "cf_cfproto"),
    ("cfrl", "cf_cfrl"),
    ("leapfactual", "cf_leapfactual"),
    ("leapfactual_r", "cf_leapfactual_r"),
    ("graco_base_o2", "cf_graco_base_o2"),
    ("graco_base_o3", "cf_graco_base_o3"),
    ("graco50", "cf_graco50"),
]

SPARSITY_EPS = 1.0 / 255.0
LPIPS_UPSIZE = 256


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    d = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).ravel()
    return float(np.sqrt(np.dot(d, d)))


def _sparsity(orig: np.ndarray, cf: np.ndarray, eps: float) -> tuple[int, float]:
    o = np.asarray(orig, dtype=np.float64).reshape(-1)
    c = np.asarray(cf, dtype=np.float64).reshape(-1)
    changed = np.abs(c - o) > eps
    n = int(changed.sum())
    return n, float(n) / float(len(o))


def _valid_cf(cf: np.ndarray) -> bool:
    a = np.asarray(cf, dtype=np.float32)
    return a.size > 0 and np.isfinite(a).any() and not np.isnan(a).all()


def _nhwc_to_01_tensor_nchw(x_nhwc: np.ndarray, device: torch.device) -> torch.Tensor:
    x = np.asarray(x_nhwc, dtype=np.float32)
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch 1, got shape {x.shape}")
        x = x[0]
    chw = np.transpose(x, (2, 0, 1))
    t = torch.tensor(chw, device=device).unsqueeze(0)
    return (t + 0.5).clamp(0, 1)


def _to_lpips_input(x01_nchw: torch.Tensor) -> torch.Tensor:
    if int(x01_nchw.shape[-1]) != LPIPS_UPSIZE or int(x01_nchw.shape[-2]) != LPIPS_UPSIZE:
        x01_nchw = F.interpolate(
            x01_nchw, size=(LPIPS_UPSIZE, LPIPS_UPSIZE), mode="bilinear", align_corners=False
        )
    return x01_nchw


def _nan_metric_fields(p_true_orig: float) -> dict[str, Any]:
    return {
        "l2_distance": math.nan,
        "sparsity_count": math.nan,
        "sparsity_frac": math.nan,
        "p_true_orig": p_true_orig,
        "p_true_cf": math.nan,
        "delta_p_true": math.nan,
        "p_intended_target_cf": math.nan,
        "pred_class_cf": math.nan,
        "p_pred_cf": math.nan,
        "inception_sqdist_to_source_class_mean": math.nan,
        "inception_sqdist_to_predicted_class_mean": math.nan,
        "lpips_to_original": math.nan,
        "lpips_to_random_pred_class": math.nan,
        "ssim_to_original": math.nan,
        "ssim_to_random_pred_class": math.nan,
    }


def _write_pngs_for_metric_dirs(dir_path: Path, images_nhwc: list[np.ndarray], prefix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dir_path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images_nhwc):
        o = np.asarray(img).reshape(299, 299, 3)
        o01 = np.clip(o + 0.5, 0.0, 1.0)
        fig, ax = plt.subplots(figsize=(1, 1), dpi=299)
        ax.imshow(o01)
        ax.set_axis_off()
        fig.savefig(dir_path / f"{prefix}_{i:04d}.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    add_imagenet_root_args(p)
    p.add_argument(
        "--classifier-weights",
        type=Path,
        default=_GRACOIMAGE_ROOT / "Imagenet" / "models" / CKPT_IMAGENET100_INCEPTION,
    )
    p.add_argument("--gen-root", type=Path, required=True)
    p.add_argument(
        "--train-pool-json",
        type=Path,
        default=_GRACOIMAGE_ROOT / "Imagenet" / "data_subset" / "train_pool_index.json",
    )
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--ref-per-class", type=int, default=32)
    p.add_argument("--batch-fid-csv", type=Path, default=None)
    args = p.parse_args()

    gen_root = args.gen_root.resolve()
    csv_out = args.csv_out or (gen_root / "per_sample_metrics.csv")
    if not gen_root.is_dir():
        raise SystemExit(f"Missing --gen-root {gen_root}")
    if not args.train_pool_json.is_file():
        raise SystemExit("Missing train_pool_index.json (run select_eval_samples).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.classifier_weights.is_file():
        raise SystemExit(f"Missing classifier {args.classifier_weights}; run train_classifier.py first.")
    clf = load_imagenet100_classifier(args.classifier_weights, device)
    inc_emb = build_inception_feature_net(device)

    pool = load_train_pool_indices(args.train_pool_json)
    cmap_path = resolve_class_map_path(args.class_map)
    ds = ImageNet100AlibiDataset(args.imagenet_root, "train", class_map_path=cmap_path)
    n_cls = int(ds._cmap["n_classes"])

    by_class: dict[int, list[np.ndarray]] = {c: [] for c in range(n_cls)}
    for pi in pool:
        img, y = ds[int(pi)]
        y = int(ds.targets[int(pi)])
        if len(by_class[y]) >= args.ref_per_class:
            continue
        by_class[y].append(np.transpose(img.numpy().astype(np.float32), (1, 2, 0)))

    print("Computing Inception class means on reference pool...")
    fdim = int(
        inception_features_from_rgb_nhwc(
            inc_emb,
            np.zeros((1, 299, 299, 3), dtype=np.float32),
            device,
            batch=1,
        ).shape[1]
    )
    mu = np.zeros((n_cls, fdim), dtype=np.float64)
    for c in range(n_cls):
        if not by_class[c]:
            continue
        x = np.stack(by_class[c], axis=0)
        f = inception_features_from_rgb_nhwc(inc_emb, x, device, batch=16)
        mu[c] = f.mean(axis=0)

    sample_dirs = sorted(gen_root.glob("sample_*"))
    if not sample_dirs:
        raise SystemExit(f"No samples under {gen_root}")

    rows: list[dict[str, Any]] = []
    by_method_records: dict[str, list[tuple[np.ndarray, int, int]]] = {m: [] for m, _ in METHOD_NPZ_KEYS}
    fieldnames = [
        "sample_idx",
        "y_true",
        "y_intended_target",
        "method",
        "has_valid_cf",
        "l2_distance",
        "sparsity_count",
        "sparsity_frac",
        "p_true_orig",
        "p_true_cf",
        "delta_p_true",
        "p_intended_target_cf",
        "pred_class_cf",
        "p_pred_cf",
        "inception_sqdist_to_source_class_mean",
        "inception_sqdist_to_predicted_class_mean",
        "lpips_to_original",
        "lpips_to_random_pred_class",
        "ssim_to_original",
        "ssim_to_random_pred_class",
    ]

    rng_ref = np.random.default_rng(0)
    lpips_metric = None
    ssim_metric = None

    for sdir in sample_dirs:
        npz_path = sdir / "tensors.npz"
        meta_path = sdir / "meta.json"
        if not npz_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        sample_idx = int(meta.get("sample_idx", sdir.name.split("_", 1)[1]))
        y_true = int(meta.get("y_true", -1))
        y_tgt = int(meta.get("y_intended_target", -1))
        if y_tgt < 0 and 0 <= y_true < n_cls:
            y_tgt = (y_true + 1) % n_cls
        z = np.load(npz_path)
        orig = np.asarray(z["orig_nhwc"], dtype=np.float32)
        if orig.ndim == 4:
            orig = orig[0]

        def _probs(x_nhwc: np.ndarray) -> np.ndarray:
            chw = np.transpose(np.asarray(x_nhwc, dtype=np.float32), (2, 0, 1))
            xt = torch.tensor(chw, device=device).unsqueeze(0)
            with torch.no_grad():
                return clf.predict_proba(xt)[0].cpu().numpy()

        p_orig = _probs(orig)
        p_true_orig = float(p_orig[y_true]) if 0 <= y_true < n_cls else math.nan

        for method, key in METHOD_NPZ_KEYS:
            if key not in z.files:
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "y_true": y_true,
                        "y_intended_target": y_tgt,
                        "method": method,
                        "has_valid_cf": 0,
                        **_nan_metric_fields(p_true_orig),
                    }
                )
                continue
            cf = np.asarray(z[key], dtype=np.float32)
            if not _valid_cf(cf):
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "y_true": y_true,
                        "y_intended_target": y_tgt,
                        "method": method,
                        "has_valid_cf": 0,
                        **_nan_metric_fields(p_true_orig),
                    }
                )
                continue
            if cf.ndim == 4:
                cf = cf[0]
            cf_nhwc = np.asarray(cf, dtype=np.float32)
            l2 = _l2(orig, cf_nhwc)
            sc, sf = _sparsity(orig, cf_nhwc, SPARSITY_EPS)
            p_cf = _probs(cf_nhwc)
            pred = int(np.argmax(p_cf))
            f_cf = inception_features_from_rgb_nhwc(inc_emb, cf_nhwc[np.newaxis, ...], device, batch=1)[0]
            d_src = float(np.sum((f_cf - mu[y_true]) ** 2)) if 0 <= y_true < n_cls else math.nan
            d_pr = float(np.sum((f_cf - mu[pred]) ** 2))

            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
                from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
            except Exception:
                lpips_o = math.nan
                lpips_r = math.nan
                ssim_o = math.nan
                ssim_r = math.nan
            else:
                if lpips_metric is None:
                    lpips_metric = LearnedPerceptualImagePatchSimilarity(
                        net_type="alex", normalize=True
                    ).to(device)
                if ssim_metric is None:
                    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

                orig01 = _nhwc_to_01_tensor_nchw(orig, device)
                cf01 = _nhwc_to_01_tensor_nchw(cf_nhwc, device)
                pool = by_class.get(pred, [])
                if pool:
                    rand_img = pool[int(rng_ref.integers(0, len(pool)))]
                    rand01 = _nhwc_to_01_tensor_nchw(rand_img, device)
                else:
                    rand01 = None

                with torch.no_grad():
                    lpips_o = float(
                        lpips_metric(_to_lpips_input(cf01), _to_lpips_input(orig01)).item()
                    )
                    ssim_o = float(ssim_metric(cf01, orig01).item())
                    if rand01 is None:
                        lpips_r = math.nan
                        ssim_r = math.nan
                    else:
                        lpips_r = float(
                            lpips_metric(_to_lpips_input(cf01), _to_lpips_input(rand01)).item()
                        )
                        ssim_r = float(ssim_metric(cf01, rand01).item())

            p_intended = (
                float(p_cf[y_tgt])
                if 0 <= y_tgt < n_cls
                else math.nan
            )

            rows.append(
                {
                    "sample_idx": sample_idx,
                    "y_true": y_true,
                    "y_intended_target": y_tgt,
                    "method": method,
                    "has_valid_cf": 1,
                    "l2_distance": l2,
                    "sparsity_count": sc,
                    "sparsity_frac": sf,
                    "p_true_orig": p_true_orig,
                    "p_true_cf": float(p_cf[y_true]) if 0 <= y_true < n_cls else math.nan,
                    "delta_p_true": float(p_cf[y_true] - p_orig[y_true]) if 0 <= y_true < n_cls else math.nan,
                    "p_intended_target_cf": p_intended,
                    "pred_class_cf": pred,
                    "p_pred_cf": float(p_cf[pred]),
                    "inception_sqdist_to_source_class_mean": d_src,
                    "inception_sqdist_to_predicted_class_mean": d_pr,
                    "lpips_to_original": lpips_o,
                    "lpips_to_random_pred_class": lpips_r,
                    "ssim_to_original": ssim_o,
                    "ssim_to_random_pred_class": ssim_r,
                }
            )
            by_method_records[method].append((cf_nhwc, y_true, pred))

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_out}")

    if args.batch_fid_csv is not None:
        try:
            from torch_fidelity import calculate_metrics
        except ImportError:
            print("torch_fidelity not installed; skip --batch-fid-csv")
        else:
            rng = np.random.default_rng(0)

            def _stratified_reals_for_method(
                records: list[tuple[np.ndarray, int, int]],
                use_pred_class: bool,
            ) -> list[np.ndarray]:
                out: list[np.ndarray] = []
                for _cf, y_true, pred in records:
                    c = pred if use_pred_class else y_true
                    pool = by_class.get(c, [])
                    if len(pool) == 0:
                        continue
                    out.append(pool[rng.integers(0, len(pool))])
                return out

            fid_rows: list[dict[str, Any]] = []
            with tempfile.TemporaryDirectory() as tmp:
                tdir = Path(tmp)
                for method, records in by_method_records.items():
                    cfs = [t[0] for t in records]
                    if len(cfs) < 2:
                        fid_rows.append(
                            {
                                "method": method,
                                "frechet_inception_distance_vs_source_y_true": math.nan,
                                "frechet_inception_distance_vs_predicted_class": math.nan,
                                "note": "need>=2 valid cfs for this method",
                            }
                        )
                        continue
                    ref_src = _stratified_reals_for_method(records, use_pred_class=False)
                    ref_pr = _stratified_reals_for_method(records, use_pred_class=True)
                    if len(ref_src) != len(cfs) or len(ref_pr) != len(cfs):
                        fid_rows.append(
                            {
                                "method": method,
                                "frechet_inception_distance_vs_source_y_true": math.nan,
                                "frechet_inception_distance_vs_predicted_class": math.nan,
                                "note": "stratified real pool missing for some classes (ref count != cf count)",
                            }
                        )
                        continue
                    src_dir = tdir / f"real_src_{method}"
                    pr_dir = tdir / f"real_pred_{method}"
                    cf_dir = tdir / f"cf_{method}"
                    _write_pngs_for_metric_dirs(src_dir, ref_src, "s")
                    _write_pngs_for_metric_dirs(pr_dir, ref_pr, "p")
                    _write_pngs_for_metric_dirs(cf_dir, cfs, "c")
                    m_src = calculate_metrics(
                        input1=str(src_dir),
                        input2=str(cf_dir),
                        cuda=device.type == "cuda",
                        fid=True,
                        verbose=False,
                    )
                    m_pr = calculate_metrics(
                        input1=str(pr_dir),
                        input2=str(cf_dir),
                        cuda=device.type == "cuda",
                        fid=True,
                        verbose=False,
                    )
                    fid_rows.append(
                        {
                            "method": method,
                            "frechet_inception_distance_vs_source_y_true": float(
                                m_src["frechet_inception_distance"]
                            ),
                            "frechet_inception_distance_vs_predicted_class": float(
                                m_pr["frechet_inception_distance"]
                            ),
                            "note": "reals stratified per-cf row: y_true vs pred_class_cf",
                        }
                    )
            args.batch_fid_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.batch_fid_csv.open("w", newline="", encoding="utf-8") as f:
                fw = csv.DictWriter(
                    f,
                    fieldnames=[
                        "method",
                        "frechet_inception_distance_vs_source_y_true",
                        "frechet_inception_distance_vs_predicted_class",
                        "note",
                    ],
                )
                fw.writeheader()
                for r in fid_rows:
                    fw.writerow(r)
            print(f"Wrote batch Fréchet Inception metrics to {args.batch_fid_csv}")


if __name__ == "__main__":
    main()
