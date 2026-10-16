"""CMMD-style + distribution Fréchet Inception Distance for ImageNet RGB counterfactuals."""
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
from PIL import Image
_GRACOIMAGE_ROOT = Path(__file__).resolve().parents[1]
_IMNET_PKG = _GRACOIMAGE_ROOT / "Imagenet"
if str(_GRACOIMAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRACOIMAGE_ROOT))
if str(_IMNET_PKG) not in sys.path:
    sys.path.insert(0, str(_IMNET_PKG))

from common.clip_cmmd import build_clip_vit_l_14_336, embed_images_rgb01
from explainers.imagenet100_inception import load_imagenet100_classifier
from imagenet_alibi_config import CKPT_IMAGENET100_INCEPTION, NUM_CLASSES

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

_SIGMA = 10.0
_SCALE = 1000.0
_CLIP_INPUT = 336


def _nhwc_alibi_to_rgb01(img_nhwc: np.ndarray) -> np.ndarray:
    x = np.asarray(img_nhwc, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    x01 = np.clip(x + 0.5, 0.0, 1.0)
    if x01.ndim == 3 and x01.shape[-1] == 3:
        u8 = (x01 * 255.0).round().astype(np.uint8)
        im = Image.fromarray(u8, mode="RGB").resize((_CLIP_INPUT, _CLIP_INPUT), Image.BICUBIC)
        return np.asarray(im, dtype=np.float32) / 255.0
    g = x01.reshape(28, 28)
    u8 = (g * 255.0).astype(np.uint8)
    im = Image.fromarray(u8, mode="L").convert("RGB").resize((_CLIP_INPUT, _CLIP_INPUT), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def _mmd_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """Gaussian-kernel MMD as in google-research cmmd/distance.py."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_sqnorms = np.sum(x * x, axis=1)
    y_sqnorms = np.sum(y * y, axis=1)
    gamma = 1.0 / (2.0 * _SIGMA**2)
    k_xx = np.exp(
        -gamma
        * (
            -2.0 * (x @ x.T)
            + x_sqnorms[:, np.newaxis]
            + x_sqnorms[np.newaxis, :]
        )
    )
    k_xy = np.exp(
        -gamma
        * (
            -2.0 * (x @ y.T)
            + x_sqnorms[:, np.newaxis]
            + y_sqnorms[np.newaxis, :]
        )
    )
    k_yy = np.exp(
        -gamma
        * (
            -2.0 * (y @ y.T)
            + y_sqnorms[:, np.newaxis]
            + y_sqnorms[np.newaxis, :]
        )
    )
    return float(_SCALE * (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()))


def _valid_cf(cf: np.ndarray) -> bool:
    a = np.asarray(cf, dtype=np.float32)
    return a.size > 0 and np.isfinite(a).any() and not np.isnan(a).all()


def _write_pngs_nhwc(dir_path: Path, images_nhwc: list[np.ndarray], prefix: str) -> None:
    """Save [-0.5, 0.5] NHWC tensors as PNG (RGB or grayscale) for torch-fidelity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dir_path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images_nhwc):
        o = np.asarray(img, dtype=np.float32)
        if o.ndim == 4:
            o = o[0]
        if o.ndim == 3 and o.shape[-1] == 3:
            import matplotlib.pyplot as plt

            o01 = np.clip(o + 0.5, 0.0, 1.0)
            plt.imsave(dir_path / f"{prefix}_{i:04d}.png", o01)
        else:
            g = o.reshape(28, 28)
            fig, ax = plt.subplots(figsize=(1, 1), dpi=28)
            ax.imshow(g, cmap="gray", vmin=-0.5, vmax=0.5)
            ax.set_axis_off()
            fig.savefig(dir_path / f"{prefix}_{i:04d}.png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)


def _nhwc_list_to_inception_01_299(
    images_nhwc: list[np.ndarray], device: torch.device
) -> torch.Tensor:
    """List of NHWC [-0.5,0.5] → (N,3,299,299) float32 in [0,1]."""
    xs = []
    for x in images_nhwc:
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 4:
            a = a[0]
        if a.ndim == 3 and a.shape[-1] == 3:
            a01 = np.clip(a + 0.5, 0.0, 1.0)
            chw = np.transpose(a01, (2, 0, 1))
        else:
            g = np.clip(a.reshape(28, 28) + 0.5, 0.0, 1.0)
            chw = np.stack([g, g, g], axis=0)
        xs.append(chw)
    t = torch.tensor(np.stack(xs, axis=0), device=device)
    if int(t.shape[-1]) != 299:
        t = F.interpolate(t, size=(299, 299), mode="bilinear", align_corners=False)
    return t.clamp(0, 1)


def _inception_01_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """torchmetrics KID + torch-fidelity Inception expect uint8 in [0, 255]."""
    return (x * 255.0).clamp(0, 255).to(torch.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="CMMD-style metrics (OpenCLIP + PyTorch) for ImageNet RGB CFs.")
    p.add_argument("--gen-root", type=Path, default=_GRACOIMAGE_ROOT / "Imagenet" / "outputs" / "generation")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to --gen-root.",
    )
    p.add_argument("--imagenet-root", type=Path, required=True)
    p.add_argument(
        "--classifier-weights",
        type=Path,
        default=_GRACOIMAGE_ROOT / "Imagenet" / "models" / CKPT_IMAGENET100_INCEPTION,
    )
    p.add_argument(
        "--train-pool-json",
        type=Path,
        default=_GRACOIMAGE_ROOT / "Imagenet" / "data_subset" / "train_pool_index.json",
    )
    p.add_argument("--ref-per-class", type=int, default=64)
    p.add_argument("--embed-batch-size", type=int, default=16)
    p.add_argument(
        "--clip-pretrained",
        type=str,
        default="openai",
        help="OpenCLIP pretrained tag for ViT-L-14-336 (default: openai).",
    )
    p.add_argument(
        "--skip-frechet",
        action="store_true",
        help="Skip Fréchet Inception distance (only CMMD-style MMD for distribution row).",
    )
    args = p.parse_args()

    gen_root = args.gen_root.resolve()
    out_dir = (args.out_dir or gen_root).resolve()

    if not gen_root.is_dir():
        raise SystemExit(f"Missing --gen-root {gen_root}")
    if not args.train_pool_json.is_file():
        raise SystemExit(f"Missing --train-pool-json {args.train_pool_json}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading OpenCLIP ViT-L-14-336 ({args.clip_pretrained}) on {device}...")
    try:
        clip_model, clip_preprocess = build_clip_vit_l_14_336(device, args.clip_pretrained)
    except ImportError as e:
        raise SystemExit(
            "Install OpenCLIP in this environment: pip install open-clip-torch"
        ) from e

    have_frechet = not args.skip_frechet
    if have_frechet:
        try:
            from torch_fidelity import calculate_metrics
        except ImportError:
            have_frechet = False
            print("Warning: torch-fidelity not installed; Fréchet column will be NaN.")

    if not args.classifier_weights.is_file():
        raise SystemExit(f"Missing classifier {args.classifier_weights}; run train_classifier.py first.")
    clf = load_imagenet100_classifier(args.classifier_weights, device, num_classes=NUM_CLASSES)

    from imagenet100_dataset import ImageNet100AlibiDataset, load_train_pool_indices
    from imagenet_data import resolve_class_map_path

    pool = load_train_pool_indices(args.train_pool_json)
    ds = ImageNet100AlibiDataset(
        args.imagenet_root, "train", class_map_path=resolve_class_map_path(None)
    )

    by_class: dict[int, list[np.ndarray]] = {c: [] for c in range(NUM_CLASSES)}
    for pi in pool:
        img, y = ds[int(pi)]
        y = int(ds.targets[int(pi)])
        if len(by_class[y]) >= args.ref_per_class:
            continue
        x_nhwc = np.transpose(img.numpy().astype(np.float32), (1, 2, 0))
        by_class[y].append(x_nhwc)

    print(f"Embedding reference pools (<= {args.ref_per_class} / class)...")
    ref_embs: dict[int, np.ndarray] = {}
    for c in range(NUM_CLASSES):
        rgb_list = [_nhwc_alibi_to_rgb01(x) for x in by_class[c]]
        if not rgb_list:
            ref_embs[c] = np.zeros((0, 0), dtype=np.float64)
            continue
        ref_embs[c] = embed_images_rgb01(
            clip_model,
            clip_preprocess,
            device,
            rgb_list,
            args.embed_batch_size,
        )

    sample_dirs = sorted(gen_root.glob("sample_*"))
    if not sample_dirs:
        raise SystemExit(f"No sample_* under {gen_root}")

    by_method_records: dict[str, list[tuple[np.ndarray, np.ndarray, int, int, int]]] = {
        m: [] for m, _ in METHOD_NPZ_KEYS
    }
    per_sample_rows: list[dict[str, Any]] = []

    for sdir in sample_dirs:
        npz_path = sdir / "tensors.npz"
        meta_path = sdir / "meta.json"
        if not npz_path.is_file():
            continue
        z = np.load(npz_path)
        orig = np.asarray(z["orig_nhwc"], dtype=np.float32)
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample_idx = int(meta.get("sample_idx", int(sdir.name.split("_", 1)[1])))
        y_true = int(meta.get("y_true", -1))
        if y_true < 0:
            continue

        for method, key in METHOD_NPZ_KEYS:
            if key not in z.files:
                per_sample_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "y_true": y_true,
                        "method": method,
                        "has_valid_cf": 0,
                        "pred_class_cf": math.nan,
                        "cmmd_vs_source_y_true_class": math.nan,
                        "cmmd_vs_predicted_class": math.nan,
                    }
                )
                continue
            cf = z[key]
            if not _valid_cf(cf):
                per_sample_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "y_true": y_true,
                        "method": method,
                        "has_valid_cf": 0,
                        "pred_class_cf": math.nan,
                        "cmmd_vs_source_y_true_class": math.nan,
                        "cmmd_vs_predicted_class": math.nan,
                    }
                )
                continue

            cf_a = np.asarray(cf, dtype=np.float32)
            if cf_a.ndim == 4:
                cf_a = cf_a[0]
            chw = np.transpose(cf_a, (2, 0, 1))
            xt = torch.tensor(chw, device=device).unsqueeze(0)
            with torch.no_grad():
                pr = clf.predict_proba(xt)[0].cpu().numpy()
            pred = int(np.argmax(pr))

            cf_rgb = [_nhwc_alibi_to_rgb01(cf)]
            cf_e = embed_images_rgb01(
                clip_model, clip_preprocess, device, cf_rgb, 1
            )
            cf_vec = cf_e[0:1]

            r_src = ref_embs.get(y_true)
            r_pr = ref_embs.get(pred)
            m_src = (
                _mmd_numpy(r_src, cf_vec)
                if r_src is not None and r_src.shape[0] >= 1 and r_src.shape[1] == cf_vec.shape[1]
                else math.nan
            )
            m_pr = (
                _mmd_numpy(r_pr, cf_vec)
                if r_pr is not None and r_pr.shape[0] >= 1 and r_pr.shape[1] == cf_vec.shape[1]
                else math.nan
            )

            per_sample_rows.append(
                {
                    "sample_idx": sample_idx,
                    "y_true": y_true,
                    "method": method,
                    "has_valid_cf": 1,
                    "pred_class_cf": pred,
                    "cmmd_vs_source_y_true_class": m_src,
                    "cmmd_vs_predicted_class": m_pr,
                }
            )
            by_method_records[method].append(
                (orig.copy(), np.asarray(cf, dtype=np.float32), y_true, pred, sample_idx)
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    ps_path = out_dir / "cmmd_per_sample.csv"
    with ps_path.open("w", newline="", encoding="utf-8") as f:
        fn = [
            "sample_idx",
            "y_true",
            "method",
            "has_valid_cf",
            "pred_class_cf",
            "cmmd_vs_source_y_true_class",
            "cmmd_vs_predicted_class",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for row in per_sample_rows:
            w.writerow(row)
    print(f"Wrote {ps_path}")

    dist_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tbase = Path(tmp_dir)

        for method, records in by_method_records.items():
            n = len(records)
            if n < 2:
                dist_rows.append(
                    {
                        "method": method,
                        "n_paired": n,
                        "frechet_inception_distance_cf_vs_originals": math.nan,
                        "kid_mean_cf_vs_originals": math.nan,
                        "kid_std_cf_vs_originals": math.nan,
                        "cmmd_orig_vs_cf_distribution": math.nan,
                        "note": "need>=2 valid cfs",
                    }
                )
                batch_rows.append(
                    {
                        "method": method,
                        "n_paired": n,
                        "cmmd_vs_stratified_source_y_true": math.nan,
                        "cmmd_vs_stratified_predicted_class": math.nan,
                        "note": "need>=2 valid cfs",
                    }
                )
                continue

            orig_list = [o for o, _cf, _yt, _pr, _ in records]
            cf_list = [cf for _o, cf, _yt, _pr, _ in records]

            fid_val = math.nan
            kid_mean = math.nan
            kid_std = math.nan
            if have_frechet:
                odir = tbase / f"fid_o_{method}"
                cdir = tbase / f"fid_c_{method}"
                _write_pngs_nhwc(odir, orig_list, "o")
                _write_pngs_nhwc(cdir, cf_list, "c")
                m_fid = calculate_metrics(
                    input1=str(odir),
                    input2=str(cdir),
                    cuda=device.type == "cuda",
                    fid=True,
                    verbose=False,
                )
                fid_val = float(m_fid["frechet_inception_distance"])

            # KID via torchmetrics (Kernel Inception Distance)
            try:
                from torchmetrics.image.kid import KernelInceptionDistance
            except Exception:
                kid_mean = math.nan
                kid_std = math.nan
            else:
                kid = KernelInceptionDistance(subset_size=min(50, n)).to(device)
                with torch.no_grad():
                    real = _inception_01_to_uint8(
                        _nhwc_list_to_inception_01_299(orig_list, device)
                    )
                    fake = _inception_01_to_uint8(
                        _nhwc_list_to_inception_01_299(cf_list, device)
                    )
                    kid.update(real, real=True)
                    kid.update(fake, real=False)
                    km, ks = kid.compute()
                kid_mean = float(km.item())
                kid_std = float(ks.item())

            orig_rgb = [_nhwc_alibi_to_rgb01(o) for o in orig_list]
            cf_rgb = [_nhwc_alibi_to_rgb01(cf) for cf in cf_list]
            e_orig = embed_images_rgb01(
                clip_model, clip_preprocess, device, orig_rgb, args.embed_batch_size
            )
            e_cf = embed_images_rgb01(
                clip_model, clip_preprocess, device, cf_rgb, args.embed_batch_size
            )
            dist_val = _mmd_numpy(e_orig, e_cf)
            dist_rows.append(
                {
                    "method": method,
                    "n_paired": n,
                    "frechet_inception_distance_cf_vs_originals": fid_val,
                    "kid_mean_cf_vs_originals": kid_mean,
                    "kid_std_cf_vs_originals": kid_std,
                    "cmmd_orig_vs_cf_distribution": dist_val,
                    "note": "paired originals vs cfs: inception fid + clip mmd",
                }
            )

            strat_src_imgs: list[np.ndarray] = []
            strat_pr_imgs: list[np.ndarray] = []
            for _o, _cf, yt, pr, _ in records:
                pool_s = by_class[yt]
                pool_p = by_class[pr]
                if pool_s:
                    strat_src_imgs.append(pool_s[rng.integers(0, len(pool_s))])
                if pool_p:
                    strat_pr_imgs.append(pool_p[rng.integers(0, len(pool_p))])

            e_strat_src = embed_images_rgb01(
                clip_model,
                clip_preprocess,
                device,
                [_nhwc_alibi_to_rgb01(x) for x in strat_src_imgs],
                args.embed_batch_size,
            )
            e_strat_pr = embed_images_rgb01(
                clip_model,
                clip_preprocess,
                device,
                [_nhwc_alibi_to_rgb01(x) for x in strat_pr_imgs],
                args.embed_batch_size,
            )
            b_src = _mmd_numpy(e_strat_src, e_cf) if e_strat_src.shape[0] >= 2 else math.nan
            b_pr = _mmd_numpy(e_strat_pr, e_cf) if e_strat_pr.shape[0] >= 2 else math.nan
            batch_rows.append(
                {
                    "method": method,
                    "n_paired": n,
                    "cmmd_vs_stratified_source_y_true": b_src,
                    "cmmd_vs_stratified_predicted_class": b_pr,
                    "note": "stratified train reals vs cf set (same n as cfs)",
                }
            )

    dpath = out_dir / "distribution_metrics.csv"
    with dpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "n_paired",
                "frechet_inception_distance_cf_vs_originals",
                "kid_mean_cf_vs_originals",
                "kid_std_cf_vs_originals",
                "cmmd_orig_vs_cf_distribution",
                "note",
            ],
        )
        w.writeheader()
        for r in dist_rows:
            w.writerow(r)
    print(f"Wrote {dpath}")

    bpath = out_dir / "cmmd_batch_stratified.csv"
    with bpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "n_paired",
                "cmmd_vs_stratified_source_y_true",
                "cmmd_vs_stratified_predicted_class",
                "note",
            ],
        )
        w.writeheader()
        for r in batch_rows:
            w.writerow(r)
    print(f"Wrote {bpath}")


if __name__ == "__main__":
    main()
