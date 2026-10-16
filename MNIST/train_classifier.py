"""Train MNISTNet; writes ``models/mnist_cnn.pt``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_MNIST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MNIST_DIR))

from alibi_mnist_config import TRAIN_ADAM_LR, TRAIN_BATCH_DEFAULT, TRAIN_EPOCHS_DEFAULT
from explainers.mnist_cnn import MNISTNet
from mnist_alibi_data import load_mnist_alibi, nhwc_to_nchw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=TRAIN_EPOCHS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=TRAIN_BATCH_DEFAULT)
    p.add_argument("--lr", type=float, default=TRAIN_ADAM_LR)
    p.add_argument("--out", type=Path, default=_MNIST_DIR / "models" / "mnist_cnn.pt")
    p.add_argument("--mnist-root", type=Path, default=_MNIST_DIR / "data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.out.is_file() and not args.force:
        print(f"Skip: {args.out} exists (use --force)")
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (x_train, y_train, _ohe_tr), (x_test, y_test, _ohe_te) = load_mnist_alibi(str(args.mnist_root))
    x_tr = torch.tensor(nhwc_to_nchw(x_train), dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    x_te = torch.tensor(nhwc_to_nchw(x_test), dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    model = MNISTNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total, correct = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = F.nll_loss(out, yb)
            loss.backward()
            opt.step()
            total += yb.size(0)
            correct += (out.argmax(dim=1) == yb).sum().item()
        print(f"Epoch {epoch}/{args.epochs} train acc {100.0 * correct / total:.2f}%")

    model.eval()
    with torch.no_grad():
        pred = model(x_te.to(device)).argmax(dim=1)
        acc = (pred == y_te.to(device)).float().mean().item()
    print(f"Test accuracy: {acc:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
