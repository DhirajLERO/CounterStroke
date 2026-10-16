"""Per-sample metrics for CIFAR-10 counterfactuals (NHWC [-0.5, 0.5])."""

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
import torch.nn as nn
import torch.nn.functional as F
_GRACOIMAGE_ROOT = Path(__file__).resolve().parents[1]
_CIFAR_PKG = _GRACOIMAGE_ROOT / "Cifar10"
if str(_CIFAR_PKG) not in sys.path:
    sys.path.insert(0, str(_CIFAR_PKG))

from cifar_alibi_data import load_cifar10_alibi, nhwc_to_nchw
from explainers.cifar_resnet import CifarResNet, conv1x1_trunk_features

SPARSITY_EPS = 1.0 / 255.0
LPIPS_UPSIZE = 64

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


@torch.no_grad()
def _backbone_features(
    model: CifarResNet,
    x_nhwc: np.ndarray,
    device: torch.device,
    batch: int = 128,
) -> np.ndarray:
    x = np.asarray(x_nhwc, dtype=np.float32)
    if x.ndim == 3:
        x = x[np.newaxis, ...]
    outs: list[np.ndarray] = []
    n = x.shape[0]
    model.eval()
    for s in range(0, n, batch):
        chunk = torch.tensor(nhwc_to_nchw(x[s : s + batch]), dtype=torch.float32, device=device)
        feat = conv1x1_trunk_features(model, chunk)
        outs.append(feat.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _per_class_means(
    x_nhwc: np.ndarray,
    y: np.ndarray,
    model: CifarResNet,
    device: torch.device,
    batch: int,
) -> tuple[np.ndarray, dict[int, int]]:
    feats = _backbone_features(model, x_nhwc, device, batch=batch)
    d = feats.shape[1]
    mu = np.zeros((10, d), dtype=np.float64)
    cnt = np.zeros(10, dtype=np.int64)
    for c in range(10):
        m = y == c
        if m.any():
            mu[c] = feats[m].mean(axis=0)
            cnt[c] = int(m.sum())
    return mu, {int(c): int(cnt[c]) for c in range(10)}


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    d = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).ravel()
    return float(np.sqrt(np.dot(d, d)))


def _sparsity(orig: np.ndarray, cf: np.ndarray, eps: float) -> tuple[int, float]:
    o = np.asarray(orig, dtype=np.float64).reshape(-1)
    c = np.asarray(cf, dtype=np.float64).reshape(-1)
    changed = np.abs(c - o) > eps
    n = int(changed.sum())
    return n, float(n) / float(len(o))


def _probs_on_nhwc(pt: CifarResNet, x_nhwc: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.tensor(nhwc_to_nchw(np.asarray(x_nhwc, dtype=np.float32)), device=device)
    with torch.no_grad():
        return pt.predict_proba(x)[0].cpu().numpy()


def _valid_cf(cf: np.ndarray) -> bool:
    a = np.asarray(cf, dtype=np.float32)
    return a.size > 0 and np.isfinite(a).any() and not np.isnan(a).all()


def _write_pngs_for_metric_dirs(dir_path: Path, images_nhwc: list[np.ndarray], prefix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dir_path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images_nhwc):
        o = np.asarray(img).reshape(32, 32, 3)
        o01 = np.clip(o + 0.5, 0.0, 1.0)
        fig, ax = plt.subplots(figsize=(1, 1), dpi=32)
        ax.imshow(o01)
        ax.set_axis_off()
        fig.savefig(dir_path / f"{prefix}_{i:04d}.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def _nhwc_to_01_tensor_nchw(x_nhwc: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.tensor(nhwc_to_nchw(np.asarray(x_nhwc, dtype=np.float32)), device=device)
    return (x + 0.5).clamp(0, 1)


def _to_lpips_input(x01_nchw: torch.Tensor) -> torch.Tensor:
    if int(x01_nchw.shape[-1]) != LPIPS_UPSIZE or int(x01_nchw.shape[-2]) != LPIPS_UPSIZE:
        x01_nchw = F.interpolate(
            x01_nchw, size=(LPIPS_UPSIZE, LPIPS_UPSIZE), mode="bilinear", align_corners=False
        )
    return x01_nchw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-root", type=Path, required=True)
    parser.add_argument(
        "--pt-weights",
        type=Path,
        default=_GRACOIMAGE_ROOT / "Cifar10" / "models" / "cifar_resnet.pt",
    )
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--cifar-root", type=Path, default=_GRACOIMAGE_ROOT / "Cifar10" / "data")
    parser.add_argument("--backbone-batch", type=int, default=256)
    parser.add_argument("--batch-fid-csv", type=Path, default=None)
    args = parser.parse_args()

    gen_root: Path = args.gen_root
    if not gen_root.is_dir():
        raise SystemExit(f"Missing --gen-root {gen_root}")

    csv_out = args.csv_out or (gen_root / "per_sample_metrics.csv")
    if not args.pt_weights.is_file():
        raise SystemExit(f"Missing --pt-weights {args.pt_weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = CifarResNet()
    pt.load_state_dict(torch.load(args.pt_weights, map_location="cpu", weights_only=True))
    pt.eval().to(device)

    (x_train, y_train, _ohe), (_xt, _yt, _ohe2) = load_cifar10_alibi(str(args.cifar_root))
    y_all = np.asarray(y_train, dtype=np.int64)
    ref_mask = np.arange(len(y_all)) >= 1000
    x_ref = x_train[ref_mask]
    y_ref = y_all[ref_mask]

    print("Building backbone class means (train idx >= 1000)...")
    mu, ref_counts = _per_class_means(x_ref, y_ref, pt, device, args.backbone_batch)
    (gen_root / "backbone_ref_counts.json").write_text(json.dumps(ref_counts, indent=2), encoding="utf-8")

    sample_dirs = sorted(gen_root.glob("sample_*"))
    if not sample_dirs:
        raise SystemExit(f"No sample_* under {gen_root}")

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

    rows: list[dict[str, Any]] = []
    by_method_records: dict[str, list[tuple[np.ndarray, int, int]]] = {m: [] for m, _ in METHOD_NPZ_KEYS}

    by_class_ref = {c: x_ref[y_ref == c] for c in range(10)}
    rng_ref = np.random.default_rng(0)

    lpips_metric = None
    ssim_metric = None

    for sdir in sample_dirs:
        print(f"Processing {sdir.name}")
        npz_path = sdir / "tensors.npz"
        meta_path = sdir / "meta.json"
        if not npz_path.is_file():
            print(f"skip {sdir.name}: no tensors.npz")
            continue
        z = np.load(npz_path)
        orig = z["orig_nhwc"]
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample_idx = int(meta.get("sample_idx", int(sdir.name.split("_", 1)[1])))
        y_true = int(meta.get("y_true", -1))
        if y_true < 0:
            raise RuntimeError(f"Bad meta in {meta_path}")
        y_intended = int(meta.get("y_intended_target", (y_true + 1) % 10))

        p_orig = _probs_on_nhwc(pt, orig, device)

        for method, key in METHOD_NPZ_KEYS:
            if key not in z.files:
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "y_true": y_true,
                        "y_intended_target": y_intended,
                        "method": method,
                        "has_valid_cf": 0,
                        "l2_distance": math.nan,
                        "sparsity_count": math.nan,
                        "sparsity_frac": math.nan,
                        "p_true_orig": float(p_orig[y_true]),
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
                )
                continue
            cf = z[key]
            ok = _valid_cf(cf)
            row: dict[str, Any] = {
                "sample_idx": sample_idx,
                "y_true": y_true,
                "y_intended_target": y_intended,
                "method": method,
                "has_valid_cf": int(ok),
            }
            if not ok:
                row.update(
                    {
                        "l2_distance": math.nan,
                        "sparsity_count": math.nan,
                        "sparsity_frac": math.nan,
                        "p_true_orig": float(p_orig[y_true]),
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
                )
                rows.append(row)
                continue

            l2v = _l2(orig, cf)
            sc, sf = _sparsity(orig, cf, SPARSITY_EPS)
            p_cf = _probs_on_nhwc(pt, cf, device)
            pred = int(np.argmax(p_cf))
            feat = _backbone_features(pt, cf, device, batch=4)[0]
            ds = float(np.sum((feat - mu[y_true]) ** 2))
            dp = float(np.sum((feat - mu[pred]) ** 2))

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
                cf01 = _nhwc_to_01_tensor_nchw(cf, device)

                pool = by_class_ref.get(pred, np.empty((0, 32, 32, 3), dtype=np.float32))
                if len(pool) > 0:
                    rand_img = pool[int(rng_ref.integers(0, len(pool)))]
                    rand01 = _nhwc_to_01_tensor_nchw(rand_img[np.newaxis, ...], device)
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

            row.update(
                {
                    "l2_distance": l2v,
                    "sparsity_count": sc,
                    "sparsity_frac": sf,
                    "p_true_orig": float(p_orig[y_true]),
                    "p_true_cf": float(p_cf[y_true]),
                    "delta_p_true": float(p_cf[y_true] - p_orig[y_true]),
                    "p_intended_target_cf": float(p_cf[y_intended]),
                    "pred_class_cf": pred,
                    "p_pred_cf": float(p_cf[pred]),
                    "inception_sqdist_to_source_class_mean": ds,
                    "inception_sqdist_to_predicted_class_mean": dp,
                    "lpips_to_original": lpips_o,
                    "lpips_to_random_pred_class": lpips_r,
                    "ssim_to_original": ssim_o,
                    "ssim_to_random_pred_class": ssim_r,
                }
            )
            rows.append(row)
            by_method_records[method].append((np.asarray(cf, dtype=np.float32), y_true, pred))

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {csv_out}")

    if args.batch_fid_csv is not None:
        try:
            from torch_fidelity import calculate_metrics
        except ImportError:
            print("torch_fidelity not installed; skip --batch-fid-csv")
        else:
            rng = np.random.default_rng(0)
            by_class = {c: x_ref[y_ref == c] for c in range(10)}

            def _stratified_reals_for_method(
                records: list[tuple[np.ndarray, int, int]],
                use_pred_class: bool,
            ) -> list[np.ndarray]:
                out: list[np.ndarray] = []
                for _cf, y_true, pred in records:
                    c = pred if use_pred_class else y_true
                    pool = by_class[c]
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
