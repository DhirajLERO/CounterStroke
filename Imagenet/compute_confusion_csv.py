"""Per-class confusion on ImageNet-100 val (100-way fine-tuned Inception) → per_class_confusion.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_IMNET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_IMNET_DIR))

from explainers.imagenet100_inception import load_imagenet100_classifier
from imagenet100_dataset import ImageNet100AlibiDataset
from imagenet_alibi_config import CKPT_IMAGENET100_INCEPTION, NUM_CLASSES
from imagenet_data import add_imagenet_root_args, resolve_class_map_path


def _write_csv(
    cm: np.ndarray,
    counts: np.ndarray,
    class_names: list[str],
    out_csv: Path,
) -> None:
    n = cm.shape[0]
    rows_out: list[dict] = []
    for c in range(n):
        denom = max(int(counts[c]), 1)
        correct = int(cm[c, c])
        acc = correct / denom
        row_cm = cm[c].copy()
        row_cm[c] = 0
        if row_cm.sum() == 0:
            tc_id_s, tc_cnt, tc_rate = "", 0, 0.0
        else:
            tc_id = int(np.argmax(row_cm))
            tc_cnt = int(row_cm[tc_id])
            tc_rate = tc_cnt / denom
            tc_id_s = str(tc_id)
        rows_out.append(
            {
                "class_id": str(c),
                "class_name": class_names[c],
                "top1_accuracy": f"{acc:.6f}",
                "top_confuser_id": tc_id_s,
                "top_confuser_count": str(tc_cnt),
                "top_confuser_rate": f"{tc_rate:.6f}",
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_id",
        "class_name",
        "top1_accuracy",
        "top_confuser_id",
        "top_confuser_count",
        "top_confuser_rate",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {len(rows_out)} rows to {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_imagenet_root_args(ap)
    ap.add_argument(
        "--classifier-weights",
        type=Path,
        default=_IMNET_DIR / "models" / CKPT_IMAGENET100_INCEPTION,
    )
    ap.add_argument("--out-csv", type=Path, default=_IMNET_DIR / "data_subset" / "per_class_confusion.csv")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if not args.classifier_weights.is_file():
        raise SystemExit(f"Missing classifier {args.classifier_weights}; run train_classifier.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cmap_path = resolve_class_map_path(args.class_map)
    ds = ImageNet100AlibiDataset(args.imagenet_root, "val", class_map_path=cmap_path)
    class_names = list(ds.class_names)
    n = NUM_CLASSES

    clf = load_imagenet100_classifier(args.classifier_weights, device, num_classes=n)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    cm = np.zeros((n, n), dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="confusion imagenet100-val"):
            xb = xb.to(device)
            pred = clf.forward_logits(xb).argmax(dim=1).cpu().numpy()
            yb_np = yb.numpy()
            for i in range(len(yb_np)):
                t = int(yb_np[i])
                p = int(pred[i])
                cm[t, p] += 1
                counts[t] += 1

    _write_csv(cm, counts, class_names, args.out_csv)


if __name__ == "__main__":
    main()
