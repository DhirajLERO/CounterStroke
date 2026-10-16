"""Pre-flight checks for data and classifier checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent
sys.path.insert(0, str(_SHARED))

from paths import BENCHMARK_ROOT, morphomnist_data_dir, celeba_data_dir  # noqa: E402
from uncond_vae_paths import resolve_uncond_vae_ckpt  # noqa: E402

MORPHO_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "train-morpho.csv",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
    "t10k-morpho.csv",
]

CELEBA_LIST_FILES = [
    "list_attr_celeba.txt",
    "list_eval_partition.txt",
]


def check_morphomnist(embeddings: str | None, uncond_ckpt: Path | None) -> list[str]:
    errs = []
    d = morphomnist_data_dir()
    for f in MORPHO_FILES:
        if not (d / f).exists() and not (d.resolve() / f).exists():
            errs.append(f"Missing MorphoMNIST file: {d / f}")
    ckpt = BENCHMARK_ROOT / "morphomnist" / "checkpoints" / "trained_classifiers"
    if not ckpt.is_dir() or not any(ckpt.iterdir()):
        errs.append(f"Missing classifier checkpoints: {ckpt}")
    if embeddings == "vae":
        resolved = uncond_ckpt or resolve_uncond_vae_ckpt(
            None, BENCHMARK_ROOT / "morphomnist", dataset="morphomnist"
        )
        if resolved is None or not resolved.is_file():
            errs.append(
                "Missing unconditional VAE for --embeddings vae: set --uncond-vae-ckpt or "
                "place .ckpt under morphomnist/checkpoints/trained_uncond_vae/"
            )
    return errs


def check_celeba(
    embeddings: str | None,
    uncond_simple: Path | None,
    uncond_complex: Path | None,
) -> list[str]:
    errs = []
    d = celeba_data_dir()
    img_dir = d / "img_align_celeba"
    if not img_dir.is_dir():
        errs.append(f"Missing CelebA images dir: {img_dir}")
    for f in CELEBA_LIST_FILES:
        if not (d / f).exists():
            errs.append(f"Missing CelebA metadata: {d / f}")
    for graph in ("simple", "complex"):
        ckpt = BENCHMARK_ROOT / "celeba" / "checkpoints" / graph / "trained_classifiers"
        if not ckpt.is_dir() or not any(ckpt.iterdir()):
            errs.append(f"Missing classifier checkpoints: {ckpt}")
    if embeddings == "vae":
        for graph, user in (("simple", uncond_simple), ("complex", uncond_complex)):
            resolved = user or resolve_uncond_vae_ckpt(
                None, BENCHMARK_ROOT / "celeba", dataset="celeba", graph=graph
            )
            if resolved is None or not resolved.is_file():
                errs.append(
                    f"Missing unconditional VAE for graph '{graph}': use "
                    f"--uncond-vae-ckpt-{graph} or checkpoints/{graph}/trained_uncond_vae/"
                )
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["morphomnist", "celeba", "both"], default="both")
    ap.add_argument("--embeddings", type=str, default=None)
    ap.add_argument("--uncond-vae-ckpt", type=Path, default=None)
    ap.add_argument("--uncond-vae-ckpt-simple", type=Path, default=None)
    ap.add_argument("--uncond-vae-ckpt-complex", type=Path, default=None)
    args = ap.parse_args()

    errs: list[str] = []
    if args.dataset in ("morphomnist", "both"):
        errs.extend(check_morphomnist(args.embeddings, args.uncond_vae_ckpt))
    if args.dataset in ("celeba", "both"):
        errs.extend(
            check_celeba(args.embeddings, args.uncond_vae_ckpt_simple, args.uncond_vae_ckpt_complex)
        )

    if errs:
        print("FAILED:")
        for e in errs:
            print(" -", e)
        raise SystemExit(1)
    print("All required assets present.")


if __name__ == "__main__":
    main()
