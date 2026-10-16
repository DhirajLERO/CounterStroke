"""Build eval_manifest (val) and train_pool_index (train) for ImageNet-100."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_IMNET_DIR = Path(__file__).resolve().parent
_ROOT = _IMNET_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_IMNET_DIR))

import torch

from common.target_class import (
    classes_best_accuracy_first,
    load_per_class_confusion_csv,
    row_for_class,
    target_class_for_row,
)
from explainers.imagenet100_inception import load_imagenet100_classifier
from imagenet100_dataset import ImageNet100AlibiDataset
from imagenet_alibi_config import CKPT_IMAGENET100_INCEPTION
from imagenet_data import add_imagenet_root_args, resolve_class_map_path


def main() -> None:
    p = argparse.ArgumentParser()
    add_imagenet_root_args(p)
    p.add_argument(
        "--classifier-weights",
        type=Path,
        default=_IMNET_DIR / "models" / CKPT_IMAGENET100_INCEPTION,
    )
    p.add_argument("--confusion-csv", type=Path, default=_IMNET_DIR / "data_subset" / "per_class_confusion.csv")
    p.add_argument("--out-json", type=Path, default=_IMNET_DIR / "data_subset" / "eval_manifest.json")
    p.add_argument(
        "--train-pool-json",
        type=Path,
        default=_IMNET_DIR / "data_subset" / "train_pool_index.json",
    )
    args = p.parse_args()

    if not args.classifier_weights.is_file():
        raise SystemExit(f"Missing classifier {args.classifier_weights}; run train_classifier.py first.")

    cmap_path = resolve_class_map_path(args.class_map)
    rows = load_per_class_confusion_csv(args.confusion_csv)
    chosen = classes_best_accuracy_first(rows, len(rows))

    ds_val = ImageNet100AlibiDataset(args.imagenet_root, "val", class_map_path=cmap_path)
    targets_val = np.asarray(ds_val.targets, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = load_imagenet100_classifier(args.classifier_weights, device)

    manifest: list[dict] = []
    for cid in chosen:
        idxs = np.flatnonzero(targets_val == int(cid))
        if idxs.size == 0:
            continue
        idx = int(idxs[0])
        img_t, y = ds_val[idx]
        x = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            pr = clf.predict_proba(x)[0].cpu().numpy()
        row = row_for_class(rows, int(cid))
        if row is None:
            raise SystemExit(f"No confusion row for class {cid}")
        tgt = target_class_for_row(row, pr)
        manifest.append(
            {
                "schema": 2,
                "split": "val",
                "sample_idx": idx,
                "dataset_index": idx,
                "y_true": int(cid),
                "y_intended_target": int(tgt),
            }
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} eval entries to {args.out_json}")

    ds_train = ImageNet100AlibiDataset(args.imagenet_root, "train", class_map_path=cmap_path)
    pool = list(range(len(ds_train)))
    payload = {
        "schema": 2,
        "split": "train",
        "indices": pool,
        "n_train_total": int(len(ds_train)),
        "n_excluded": 0,
    }
    args.train_pool_json.parent.mkdir(parents=True, exist_ok=True)
    args.train_pool_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote train pool ({len(pool)} indices) to {args.train_pool_json}")


if __name__ == "__main__":
    main()
