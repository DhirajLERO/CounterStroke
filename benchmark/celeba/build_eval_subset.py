"""Build eval_indices and eval_interventions JSON for CelebA (per graph)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT.parent / "shared"))
from paths import setup_vendor_path  # noqa: E402

setup_vendor_path()
from datasets.transforms import ReturnDictTransform  # noqa: E402
from datasets.celeba.dataset import Celeba  # noqa: E402
from eval_subset import build_eval_indices, celeba_simple_stratify, celeba_complex_stratify  # noqa: E402
from eval_interventions import build_intervention_manifest  # noqa: E402
from eval_protocol import DEFAULT_EVAL_SAMPLES, interventions_path_for_indices  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=["simple", "complex"], default="simple")
    ap.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = json.loads((_ROOT / "configs" / args.graph / "vae.json").read_text())
    attribute_size = cfg["attribute_size"]
    t = ReturnDictTransform(attribute_size)
    ds = Celeba(attribute_size, split="test", transform=t, data_dir=str(_ROOT / "data"))
    out = _ROOT / "data_subset" / f"eval_indices_{args.graph}.json"
    stratify = celeba_simple_stratify(ds) if args.graph == "simple" else celeba_complex_stratify(ds)
    per_stratum = max(1, args.eval_samples // 2)
    indices = build_eval_indices(
        ds,
        n_indices=args.eval_samples,
        per_stratum=per_stratum,
        seed=args.seed,
        out_path=out,
        stratify_fn=stratify,
    )
    manifest_path = interventions_path_for_indices(out)
    build_intervention_manifest(
        ds,
        indices,
        attribute_size,
        seed=args.seed,
        out_path=manifest_path,
        transform=t,
    )
    print(f"Wrote {out} ({len(indices)} indices)")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
