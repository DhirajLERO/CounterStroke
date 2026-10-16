"""Fine-tune InceptionV3 on ImageNet-100 train; writes ``models/imagenet100_inception.pt``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_IMNET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_IMNET_DIR))

from explainers.imagenet100_inception import build_imagenet100_inception
from imagenet100_dataset import ImageNet100AlibiDataset
from imagenet_alibi_config import (
    CKPT_IMAGENET100_INCEPTION,
    CLASSIFIER_TRAIN_BATCH,
    CLASSIFIER_TRAIN_EPOCHS,
    CLASSIFIER_TRAIN_LR,
    NUM_CLASSES,
)
from imagenet_data import add_imagenet_root_args, resolve_class_map_path


def main() -> None:
    p = argparse.ArgumentParser()
    add_imagenet_root_args(p)
    p.add_argument("--epochs", type=int, default=CLASSIFIER_TRAIN_EPOCHS)
    p.add_argument("--batch-size", type=int, default=CLASSIFIER_TRAIN_BATCH)
    p.add_argument("--lr", type=float, default=CLASSIFIER_TRAIN_LR)
    p.add_argument(
        "--out",
        type=Path,
        default=_IMNET_DIR / "models" / CKPT_IMAGENET100_INCEPTION,
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.out.is_file() and not args.force:
        print(f"Skip: {args.out} exists (use --force)")
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cmap_path = resolve_class_map_path(args.class_map)

    train_ds = ImageNet100AlibiDataset(args.imagenet_root, "train", class_map_path=cmap_path)
    val_ds = ImageNet100AlibiDataset(args.imagenet_root, "val", class_map_path=cmap_path)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_ds) > args.batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_imagenet100_inception(NUM_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct = 0, 0
        for xb, yb in tqdm(train_loader, desc=f"train e{epoch}/{args.epochs}", leave=False):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.nll_loss(model(xb), yb)
            loss.backward()
            opt.step()
            total += yb.size(0)
            correct += (model.forward_logits(xb).argmax(dim=1) == yb).sum().item()
        train_acc = 100.0 * correct / max(total, 1)

        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                pred = model.forward_logits(xb).argmax(dim=1)
                v_total += yb.size(0)
                v_correct += (pred == yb).sum().item()
        val_acc = 100.0 * v_correct / max(v_total, 1)
        print(f"Epoch {epoch}/{args.epochs} train_acc={train_acc:.2f}% val_acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  saved best checkpoint → {args.out}")

    if best_val_acc < 0:
        torch.save(model.state_dict(), args.out)
    print(f"Done. Best val_acc={best_val_acc:.2f}% → {args.out}")


if __name__ == "__main__":
    main()
