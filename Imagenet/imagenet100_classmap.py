"""ImageNet-100 WordNet folder ids + Kaggle Labels.json → local class index 0..99."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_DEFAULT_LABELS_NAME = "imagenet100_labels.json"


def primary_class_name(full_label: str) -> str:
    """First synonym from Kaggle label string (e.g. ``'tench, Tinca tinca'`` → ``'tench'``)."""
    return str(full_label).split(",")[0].strip()


def load_kaggle_labels(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Labels file must be a JSON object: {path}")
    return {str(k): str(v) for k, v in raw.items()}


def default_labels_path() -> Path:
    return Path(__file__).resolve().parent / "data_subset" / _DEFAULT_LABELS_NAME


def resolve_labels_path(explicit: Path | None) -> Path:
    return explicit if explicit is not None else default_labels_path()


def find_labels_json(cache_root: Path) -> Path | None:
    """Locate ``Labels.json`` next to Kaggle shards (``train.X*``, ``val.X``)."""
    root = cache_root.resolve()
    for name in ("Labels.json", "labels.json"):
        p = root / name
        if p.is_file():
            return p
    return None


def sorted_class_folders(split_dir: Path) -> list[str]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    names = [p.name for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not names:
        raise FileNotFoundError(f"No class folders under {split_dir}")
    return sorted(names)


def build_class_map(imagenet100_root: Path, labels_path: Path | None = None) -> dict:
    """Build schema-2 class map: sorted WordNet folders + human-readable names from Labels.json."""
    root = Path(imagenet100_root).resolve()
    labels_p = resolve_labels_path(labels_path)
    if not labels_p.is_file():
        raise FileNotFoundError(
            f"Missing {labels_p}; run download_imagenet100.py or pass --labels-json"
        )
    labels = load_kaggle_labels(labels_p)

    train_dir = root / "train"
    val_dir = root / "val"
    train_folders = sorted_class_folders(train_dir)
    val_folders = sorted_class_folders(val_dir)
    if set(train_folders) != set(val_folders):
        raise ValueError(
            f"train/val class folders differ: "
            f"train-only={set(train_folders) - set(val_folders)}, "
            f"val-only={set(val_folders) - set(train_folders)}"
        )

    wnids: list[str] = []
    class_names: list[str] = []
    labels_full: dict[str, str] = {}
    for wnid in train_folders:
        if wnid not in labels:
            raise KeyError(f"WordNet id {wnid!r} missing from {labels_p}")
        full = labels[wnid]
        labels_full[wnid] = full
        wnids.append(wnid)
        class_names.append(primary_class_name(full))

    return {
        "schema": 2,
        "n_classes": len(wnids),
        "wnids": wnids,
        "class_names": class_names,
        "labels_full": labels_full,
    }


def default_class_map_path() -> Path:
    return Path(__file__).resolve().parent / "data_subset" / "imagenet100_class_map.json"


def resolve_class_map_path(explicit: Path | None) -> Path:
    return explicit if explicit is not None else default_class_map_path()


def load_class_map(path: Path | None = None) -> dict:
    p = path or default_class_map_path()
    if not p.is_file():
        raise FileNotFoundError(f"Missing class map {p}; run download_imagenet100.py first")
    cmap = json.loads(p.read_text(encoding="utf-8"))
    if int(cmap.get("schema", 1)) < 2:
        raise ValueError(
            f"Outdated class map schema in {p}; re-run download_imagenet100.py to regenerate."
        )
    return cmap


def class_index_for_wnid(class_map: dict, wnid: str) -> int:
    try:
        return class_map["wnids"].index(str(wnid))
    except ValueError as e:
        raise KeyError(f"Unknown wnid {wnid!r}") from e


def copy_labels_from_cache(cache_root: Path, dest: Path) -> Path:
    src = find_labels_json(cache_root)
    if src is None:
        raise FileNotFoundError(f"No Labels.json under {cache_root}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest
