"""Select 100 CIFAR-10 test samples (10 per class) + targets; write train pool JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

_CIFAR_DIR = Path(__file__).resolve().parent
_ROOT = _CIFAR_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_CIFAR_DIR))

from cifar_alibi_config import CKPT_CIFAR_RESNET
from cifar_alibi_data import nhwc_to_nchw
from common.target_class import load_per_class_confusion_csv, row_for_class, target_class_for_row
from explainers.cifar_resnet import CifarResNet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--confusion-csv", type=Path, default=_CIFAR_DIR / "data_subset" / "per_class_confusion.csv")
    p.add_argument("--cifar-root", type=Path, default=_CIFAR_DIR / "data")
    p.add_argument("--per-class", type=int, default=10)
    p.add_argument("--out-json", type=Path, default=_CIFAR_DIR / "data_subset" / "eval_manifest.json")
    p.add_argument(
        "--train-pool-json",
        type=Path,
        default=_CIFAR_DIR / "data_subset" / "train_pool_index.json",
    )
    p.add_argument("--pt-weights", type=Path, default=_CIFAR_DIR / "models" / CKPT_CIFAR_RESNET)
    args = p.parse_args()

    rows = load_per_class_confusion_csv(args.confusion_csv)
    ds = CIFAR10(root=str(args.cifar_root), train=False, download=True)
    y_all = np.asarray(ds.targets, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = CifarResNet().to(device)
    pt.load_state_dict(torch.load(args.pt_weights, map_location="cpu", weights_only=True))
    pt.eval()

    manifest: list[dict] = []
    for c in range(10):
        idxs = np.flatnonzero(y_all == c)
        idxs = np.sort(idxs)[: args.per_class]
        row = row_for_class(rows, c)
        if row is None:
            raise SystemExit(f"No confusion row for class {c}")
        for idx in idxs:
            img_arr, _ = ds.data[int(idx)], ds.targets[int(idx)]
            x = (np.asarray(img_arr, dtype=np.float32) / 255.0).astype(np.float32)
            lo, hi = -0.5, 0.5
            xmin, xmax = float(x.min()), float(x.max())
            x = ((x - xmin) / (xmax - xmin + 1e-12)) * (hi - lo) + lo
            xt = torch.tensor(nhwc_to_nchw(x[np.newaxis, ...]), device=device)
            with torch.no_grad():
                probs = pt.predict_proba(xt)[0].cpu().numpy()
            tgt = target_class_for_row(row, probs)
            manifest.append(
                {
                    "sample_idx": int(idx),
                    "split": "test",
                    "y_true": c,
                    "y_intended_target": int(tgt),
                }
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} entries to {args.out_json}")

    train_ds = CIFAR10(root=str(args.cifar_root), train=True, download=True)
    n_train = len(train_ds)
    payload = {
        "indices": list(range(n_train)),
        "n_train_total": int(n_train),
        "n_excluded": 0,
    }
    args.train_pool_json.parent.mkdir(parents=True, exist_ok=True)
    args.train_pool_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote train pool ({n_train} indices) to {args.train_pool_json}")


if __name__ == "__main__":
    main()
