"""Compute per-class confusion stats on MNIST test → ``data_subset/per_class_confusion.csv``."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

_MNIST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MNIST_DIR))
_ROOT = _MNIST_DIR.parent
sys.path.insert(0, str(_ROOT))

from explainers.mnist_cnn import MNISTNet
from mnist_alibi_data import load_mnist_alibi, nhwc_to_nchw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pt-weights", type=Path, default=_MNIST_DIR / "models" / "mnist_cnn.pt")
    p.add_argument("--mnist-root", type=Path, default=_MNIST_DIR / "data")
    p.add_argument("--out-csv", type=Path, default=_MNIST_DIR / "data_subset" / "per_class_confusion.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = MNISTNet().to(device)
    pt.load_state_dict(torch.load(args.pt_weights, map_location="cpu", weights_only=True))
    pt.eval()

    (_x_tr, _y_tr, _ohe_tr), (x_te, y_te, _ohe_te) = load_mnist_alibi(str(args.mnist_root))
    x_nchw = torch.tensor(nhwc_to_nchw(x_te), dtype=torch.float32, device=device)
    y_te = np.asarray(y_te, dtype=np.int64)

    cm = np.zeros((10, 10), dtype=np.int64)
    counts = np.zeros(10, dtype=np.int64)
    with torch.no_grad():
        preds = pt(x_nchw).argmax(dim=1).cpu().numpy()

    for i in range(len(y_te)):
        t, pr = int(y_te[i]), int(preds[i])
        counts[t] += 1
        cm[t, pr] += 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_id",
        "class_name",
        "top1_accuracy",
        "top_confuser_id",
        "top_confuser_count",
        "top_confuser_rate",
    ]
    rows_out = []
    for c in range(10):
        denom = max(int(counts[c]), 1)
        correct = int(cm[c, c])
        acc = correct / denom
        row_cm = cm[c].copy()
        row_cm[c] = 0
        if row_cm.sum() == 0:
            tc_id, tc_cnt = "", 0
            tc_rate = 0.0
        else:
            tc_id = int(np.argmax(row_cm))
            tc_cnt = int(row_cm[tc_id])
            tc_rate = tc_cnt / denom
        rows_out.append(
            {
                "class_id": f"{c:05d}",
                "class_name": str(c),
                "top1_accuracy": f"{acc:.6f}",
                "top_confuser_id": f"{tc_id:05d}" if tc_cnt > 0 else "",
                "top_confuser_count": str(tc_cnt),
                "top_confuser_rate": f"{tc_rate:.6f}",
            }
        )

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
