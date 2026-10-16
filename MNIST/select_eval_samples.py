"""Select MNIST test samples and write the disjoint train-split pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST

_MNIST_DIR = Path(__file__).resolve().parent
_ROOT = _MNIST_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_MNIST_DIR))

from common.target_class import load_per_class_confusion_csv, row_for_class, target_class_for_row
from explainers.mnist_cnn import MNISTNet
from mnist_alibi_data import nhwc_to_nchw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--confusion-csv", type=Path, default=_MNIST_DIR / "data_subset" / "per_class_confusion.csv")
    p.add_argument("--mnist-root", type=Path, default=_MNIST_DIR / "data")
    p.add_argument("--per-class", type=int, default=10)
    p.add_argument("--out-json", type=Path, default=_MNIST_DIR / "data_subset" / "eval_manifest.json")
    p.add_argument(
        "--train-pool-json",
        type=Path,
        default=_MNIST_DIR / "data_subset" / "train_pool_index.json",
    )
    p.add_argument("--pt-weights", type=Path, default=_MNIST_DIR / "models" / "mnist_cnn.pt")
    args = p.parse_args()

    rows = load_per_class_confusion_csv(args.confusion_csv)
    ds = MNIST(root=str(args.mnist_root), train=False, download=True, transform=None)
    y_all = ds.targets.numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = MNISTNet().to(device)
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
            img_t, _ = ds[int(idx)]
            x = (np.asarray(img_t, dtype=np.float32) / 255.0).reshape(28, 28, 1).astype(np.float32) - 0.5
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

    train_ds = MNIST(root=str(args.mnist_root), train=True, download=True, transform=None)
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
