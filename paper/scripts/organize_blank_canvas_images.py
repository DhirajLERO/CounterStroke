#!/usr/bin/env python3
"""Rename blank-canvas PNGs and copy into paper/figures/blank_canvas/."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "blankcanvas images"
OUT = ROOT / "paper" / "figures" / "blank_canvas"

CIFAR_CLASSES = [
    "automobile", "bird", "cat", "deer", "dog",
    "frog", "horse", "ship", "truck",
]

IMAGENET = [
    ("cock", "cock"),
    ("electricray", "electric_ray"),
    ("goldfish", "goldfish"),
    ("tench", "tench"),
    ("tigershark", "tiger_shark"),
]


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _rename_src(old: Path, new: Path) -> Path:
    if not old.exists():
        return new if new.exists() else old
    if old == new:
        return old
    if new.exists():
        new.unlink()
    old.rename(new)
    return new


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing source folder: {SRC}")

    for digit in range(10):
        cf_old = SRC / f"mnist{digit}.png"
        att_old = SRC / f"mnist{digit}_cochange.png"
        cf_name = f"mnist_digit{digit}_cf.png"
        att_name = f"mnist_digit{digit}_cochange.png"
        cf_src = _rename_src(cf_old, SRC / cf_name)
        att_src = _rename_src(att_old, SRC / att_name)
        _copy(cf_src, OUT / "mnist" / cf_name)
        _copy(att_src, OUT / "mnist" / att_name)

    for cls in CIFAR_CLASSES:
        cf_old = SRC / f"cifar_{cls}.png"
        att_old = SRC / f"cifar_{cls}_cochange.png"
        cf_name = f"cifar10_{cls}_cf.png"
        att_name = f"cifar10_{cls}_cochange.png"
        cf_src = _rename_src(cf_old, SRC / cf_name)
        att_src = _rename_src(att_old, SRC / att_name)
        _copy(cf_src, OUT / "cifar10" / cf_name)
        _copy(att_src, OUT / "cifar10" / att_name)

    for old_slug, new_slug in IMAGENET:
        cf_old = SRC / f"imagenet_{old_slug}.png"
        att_old = SRC / f"imagenet_{old_slug}_cochange.png"
        if old_slug == "electricray" and not att_old.exists():
            att_old = SRC / "imagenet_electricraycochange.png"
        cf_name = f"imagenet100_{new_slug}_cf.png"
        att_name = f"imagenet100_{new_slug}_cochange.png"
        cf_src = _rename_src(cf_old, SRC / cf_name)
        att_src = _rename_src(att_old, SRC / att_name)
        _copy(cf_src, OUT / "imagenet100" / cf_name)
        _copy(att_src, OUT / "imagenet100" / att_name)

    print(f"Organized images under {OUT}")
    for sub in ("mnist", "cifar10", "imagenet100"):
        d = OUT / sub
        if d.exists():
            print(f"  {sub}: {len(list(d.glob('*.png')))} files")


if __name__ == "__main__":
    main()
