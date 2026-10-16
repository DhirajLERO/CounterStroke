"""Download ImageNet-100 from Kaggle and normalize to train/val ImageFolder layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

_IMNET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_IMNET_DIR))

from imagenet100_classmap import (
    build_class_map,
    copy_labels_from_cache,
    default_labels_path,
    find_labels_json,
)

_TRAIN_SHARD_RE = re.compile(r"^train\.x\d+$", re.IGNORECASE)
_VAL_DIR_RE = re.compile(r"^val(\.x\d*)?$", re.IGNORECASE)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_class_dir(p: Path) -> bool:
    return p.is_dir() and not p.name.startswith(".")


def _iter_images(class_dir: Path):
    for p in sorted(class_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            yield p


def _detect_layout(downloaded: Path) -> tuple[list[Path], Path]:
    """Return (train_shards, val_root)."""
    root = downloaded.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    children = [p for p in root.iterdir() if p.is_dir()]

    train_std = root / "train"
    val_std = root / "val"
    if train_std.is_dir() and val_std.is_dir():
        return [train_std], val_std

    train_shards = sorted(
        (p for p in children if _TRAIN_SHARD_RE.match(p.name)),
        key=lambda p: p.name.lower(),
    )
    val_dirs = [p for p in children if _VAL_DIR_RE.match(p.name)]
    if train_shards and val_dirs:
        val_root = min(val_dirs, key=lambda p: p.name.lower())
        return train_shards, val_root

    # Fallback: search nested trees (older caches or repacked archives)
    candidates_train: list[Path] = []
    candidates_val: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        low = p.name.lower()
        if low == "train":
            candidates_train.append(p)
        elif low in ("val", "validation", "valid") or low.startswith("val."):
            candidates_val.append(p)
        elif _TRAIN_SHARD_RE.match(p.name):
            candidates_train.append(p)

    shard_train = sorted(
        (p for p in candidates_train if _TRAIN_SHARD_RE.match(p.name)),
        key=lambda p: (len(p.parts), p.name.lower()),
    )
    if shard_train:
        val_pool = [p for p in candidates_val if _VAL_DIR_RE.match(p.name)]
        if val_pool:
            return shard_train, min(val_pool, key=lambda p: len(p.parts))

    if candidates_train and candidates_val:
        train_root = min(candidates_train, key=lambda p: len(p.parts))
        val_root = min(candidates_val, key=lambda p: len(p.parts))
        return [train_root], val_root

    listing = ", ".join(sorted(p.name for p in children)) or "(empty)"
    raise FileNotFoundError(
        f"Could not find train/val or train.X*/val.X under {root}. "
        f"Top-level entries: {listing}"
    )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _link_or_copy_file(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _materialize_train_shards(shards: list[Path], dst: Path, *, copy: bool) -> None:
    """Merge one or more train shards into ``dst/<wnid>/`` ImageFolder layout."""
    if dst.exists():
        _remove_path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    if len(shards) == 1 and all(_is_class_dir(p) for p in shards[0].iterdir()):
        shard = shards[0]
        if copy:
            shutil.copytree(shard, dst, dirs_exist_ok=True)
        else:
            dst.symlink_to(shard.resolve())
        return

    n_images = 0
    n_classes: set[str] = set()
    for shard in shards:
        for class_dir in sorted(p for p in shard.iterdir() if _is_class_dir(p)):
            wnid = class_dir.name
            n_classes.add(wnid)
            out_class = dst / wnid
            out_class.mkdir(parents=True, exist_ok=True)
            for img in _iter_images(class_dir):
                out_name = img.name
                if (out_class / out_name).exists():
                    stem, suf = img.stem, img.suffix
                    out_name = f"{stem}_{shard.name}{suf}"
                _link_or_copy_file(img, out_class / out_name, copy=copy)
                n_images += 1
    print(f"  train: {len(shards)} shard(s), {len(n_classes)} classes, {n_images} images → {dst}")


def _materialize_val(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists():
        _remove_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.symlink_to(src.resolve())
    n_classes = sum(1 for p in src.iterdir() if _is_class_dir(p))
    print(f"  val: {n_classes} classes from {src} → {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        type=Path,
        default=_IMNET_DIR / "data" / "imagenet100",
        help="Target ImageFolder root with train/ and val/",
    )
    ap.add_argument(
        "--class-map-out",
        type=Path,
        default=_IMNET_DIR / "data_subset" / "imagenet100_class_map.json",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking from kagglehub cache",
    )
    ap.add_argument(
        "--labels-json",
        type=Path,
        default=None,
        help="Kaggle Labels.json (default: data_subset/imagenet100_labels.json)",
    )
    ap.add_argument("--skip-kaggle", action="store_true", help="Only build class map if layout exists")
    args = ap.parse_args()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    labels_dest = args.labels_json if args.labels_json is not None else default_labels_path()

    if not args.skip_kaggle:
        try:
            import kagglehub
        except ImportError as e:
            raise SystemExit("Install kagglehub: pip install kagglehub") from e

        print("Downloading ambityga/imagenet100 via kagglehub...")
        cache_path = Path(kagglehub.dataset_download("ambityga/imagenet100"))
        print(f"Kaggle cache: {cache_path}")
        train_shards, val_src = _detect_layout(cache_path)
        print(f"Detected layout: {len(train_shards)} train shard(s), val={val_src.name}")
        _materialize_train_shards(train_shards, out_root / "train", copy=args.copy)
        _materialize_val(val_src, out_root / "val", copy=args.copy)
        labels_src = find_labels_json(cache_path)
        if labels_src is not None:
            copy_labels_from_cache(cache_path, labels_dest)
            print(f"Copied labels to {labels_dest}")
        else:
            print(f"Warning: no Labels.json under {cache_path}; use existing {labels_dest}")

    if not (out_root / "train").is_dir() or not (out_root / "val").is_dir():
        raise SystemExit(f"Missing train/ or val/ under {out_root}")

    class_map = build_class_map(out_root, labels_path=labels_dest)
    args.class_map_out.parent.mkdir(parents=True, exist_ok=True)
    args.class_map_out.write_text(json.dumps(class_map, indent=2), encoding="utf-8")
    print(f"Wrote class map ({class_map['n_classes']} classes) to {args.class_map_out}")

    sentinel = out_root / ".download_done"
    sentinel.write_text("ok\n", encoding="utf-8")
    print(f"Done. Sentinel: {sentinel}")


if __name__ == "__main__":
    main()
