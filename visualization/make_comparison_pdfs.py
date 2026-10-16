#!/usr/bin/env python3
"""Build per-sample comparison PDFs."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import deque
from pathlib import Path

# Allow running as script: import sibling module
_VIZ = Path(__file__).resolve().parent
if str(_VIZ) not in sys.path:
    sys.path.insert(0, str(_VIZ))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _plotting import (
    METHOD_COLUMNS,
    load_tensors_npz,
    parse_sample_dir_idx,
    reference_hw_from_orig,
    tensor_to_panel_hwc,
)


_GRACO_IMAGE = _VIZ.parent

DEFAULT_GEN = {
    "cifar10": _GRACO_IMAGE / "Cifar10" / "outputs" / "generation",
    "imagenet": _GRACO_IMAGE / "Imagenet" / "outputs" / "generation",
    "mnist": _GRACO_IMAGE / "MNIST" / "outputs" / "generation",
}


def _discover_sample_dirs(gen_root: Path) -> list[Path]:
    if not gen_root.is_dir():
        return []
    dirs = [p for p in gen_root.iterdir() if p.is_dir() and p.name.startswith("sample_")]
    dirs.sort(key=lambda p: parse_sample_dir_idx(p))
    return dirs


def _group_by_true_class(dirs: list[Path]) -> tuple[dict[int, deque[Path]], int]:
    """Returns ({class_id: deque of sample dirs sorted by idx}), n_skipped_missing_meta)."""
    raw: dict[int, list[Path]] = {}
    skipped = 0
    for d in dirs:
        mp = d / "meta.json"
        if not mp.is_file():
            skipped += 1
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
            if "y_true" not in meta:
                skipped += 1
                continue
            y = int(meta["y_true"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            skipped += 1
            continue
        raw.setdefault(y, []).append(d)

    pools: dict[int, deque[Path]] = {}
    for y in sorted(raw.keys()):
        lst = sorted(raw[y], key=lambda p: parse_sample_dir_idx(p))
        pools[y] = deque(lst)
    return pools, skipped


def _round_robin_pick(pools: dict[int, deque[Path]], n: int) -> list[Path]:
    if n <= 0:
        return []
    classes = sorted(pools.keys())
    out: list[Path] = []
    while len(out) < n:
        progressed = False
        for c in classes:
            if len(out) >= n:
                break
            if pools[c]:
                out.append(pools[c].popleft())
                progressed = True
        if not progressed:
            break
    return out


def _warn_coverage(*, n_pick: int, n_classes_with_data: int, total_samples: int) -> None:
    if n_pick > total_samples:
        warnings.warn(
            f"Requested {n_pick} PDFs but only {total_samples} sample_* folders "
            f"(after grouping); fewer PDFs will be written.",
            stacklevel=2,
        )
    if n_classes_with_data > 0 and n_pick < n_classes_with_data:
        warnings.warn(
            f"Requested {n_pick} PDFs but {n_classes_with_data} classes have samples — "
            "cannot include every class at least once.",
            stacklevel=2,
        )


def _render_pdf_for_sample(
    sample_dir: Path,
    out_pdf: Path,
    *,
    dpi: float,
) -> None:
    npz_path = sample_dir / "tensors.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing {npz_path}")

    tensors = load_tensors_npz(npz_path)
    orig = tensors.get("orig_nhwc")
    ref_hw = reference_hw_from_orig(orig)

    n_cols = len(METHOD_COLUMNS)
    fig_w = max(2.0 * n_cols, 6.0)
    fig_h = 3.2
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
    if n_cols == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, METHOD_COLUMNS):
        arr = tensors.get(key)
        rgb, ok = tensor_to_panel_hwc(arr, ref_hw=ref_hw)
        ax.imshow(rgb, vmin=0.0, vmax=1.0)
        ttl = title if ok else f"{title}\n(N/A)"
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")

    fig.suptitle(sample_dir.name, fontsize=11, y=1.02)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _process_dataset(
    name: str,
    gen_root: Path,
    out_root: Path,
    *,
    num_examples: int,
    dpi: float,
) -> int:
    dirs_all = _discover_sample_dirs(gen_root)
    if not dirs_all:
        print(f"[{name}] No sample_* under {gen_root} — skip.", file=sys.stderr)
        return 0

    pools, skipped = _group_by_true_class(dirs_all)
    if not pools:
        print(
            f"[{name}] No usable meta.json with y_true ({skipped} dirs skipped) — skip.",
            file=sys.stderr,
        )
        return 0

    total_samples = sum(len(d) for d in pools.values())
    n_classes = len(pools)
    _warn_coverage(
        n_pick=num_examples,
        n_classes_with_data=n_classes,
        total_samples=total_samples,
    )

    # Copy pools for destructive round-robin
    pools_use = {c: deque(list(q)) for c, q in sorted(pools.items())}
    chosen = _round_robin_pick(pools_use, num_examples)
    if not chosen:
        return 0

    ds_out = out_root / name
    ds_out.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for d in chosen:
        pdf_path = ds_out / f"{d.name}.pdf"
        try:
            _render_pdf_for_sample(d, pdf_path, dpi=dpi)
            n_ok += 1
        except Exception as ex:  # noqa: BLE001
            print(f"[{name}] Failed {d.name}: {ex}", file=sys.stderr)
    print(f"[{name}] Wrote {n_ok} PDF(s) to {ds_out}")
    return n_ok


def _parse_datasets(arg: str | None) -> list[str]:
    if not arg or not arg.strip():
        return ["cifar10", "imagenet", "mnist"]
    parts = [p.strip().lower() for p in arg.split(",") if p.strip()]
    valid = {"cifar10", "imagenet", "mnist"}
    bad = [p for p in parts if p not in valid]
    if bad:
        raise SystemExit(f"Unknown --datasets entries: {bad}; use {sorted(valid)}")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export horizontal comparison PDFs per sample (Original + CF methods)."
    )
    ap.add_argument(
        "--num-examples",
        type=int,
        required=True,
        help="PDFs per dataset (class-balanced round-robin over sample_* folders).",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=_VIZ / "out",
        help="Directory for cifar10/, imagenet/, mnist/ PDF subfolders.",
    )
    ap.add_argument(
        "--cifar-gen",
        type=Path,
        default=DEFAULT_GEN["cifar10"],
        help="CIFAR-10 outputs/generation root.",
    )
    ap.add_argument(
        "--imagenet-gen",
        type=Path,
        default=DEFAULT_GEN["imagenet"],
        help="ImageNet outputs/generation root.",
    )
    ap.add_argument(
        "--mnist-gen",
        type=Path,
        default=DEFAULT_GEN["mnist"],
        help="MNIST outputs/generation root.",
    )
    ap.add_argument("--dpi", type=float, default=300.0, help="Figure DPI (default 300).")
    ap.add_argument(
        "--datasets",
        type=str,
        default="cifar10,imagenet,mnist",
        help="Comma-separated subset: cifar10,imagenet,mnist.",
    )
    args = ap.parse_args()

    if args.num_examples < 1:
        raise SystemExit("--num-examples must be >= 1")

    datasets = _parse_datasets(args.datasets)
    roots = {
        "cifar10": args.cifar_gen.resolve(),
        "imagenet": args.imagenet_gen.resolve(),
        "mnist": args.mnist_gen.resolve(),
    }
    out_root = args.output_root.resolve()

    total = 0
    for ds in datasets:
        total += _process_dataset(ds, roots[ds], out_root, num_examples=args.num_examples, dpi=args.dpi)
    print(f"Done. Total PDFs written: {total}")


if __name__ == "__main__":
    main()
