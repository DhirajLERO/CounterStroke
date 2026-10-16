"""Download / cache Inception V3 weights under ``Imagenet/models/`` (sentinel for pipeline)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.models import Inception_V3_Weights, inception_v3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "inception_v3_imagenet1k_v1.pt",
    )
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    m.aux_logits = False
    torch.save(m.state_dict(), args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
