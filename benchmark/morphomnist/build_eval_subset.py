"""Build eval_indices.json and eval_interventions.json for MorphoMNIST."""

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
from datasets.morphomnist.dataset import MorphoMNISTLike  # noqa: E402
from eval_subset import build_eval_indices, morphomnist_stratify  # noqa: E402
from eval_interventions import build_intervention_manifest  # noqa: E402
from eval_protocol import DEFAULT_EVAL_SAMPLES, interventions_path_for_indices  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = json.loads((_ROOT / "configs" / "vae.json").read_text())
    attribute_size = cfg["attribute_size"]
    t = ReturnDictTransform(attribute_size)
    ds = MorphoMNISTLike(attribute_size, split="test", transform=t)
    out = _ROOT / "data_subset" / "eval_indices.json"
    per_stratum = max(1, args.eval_samples // 10)
    indices = build_eval_indices(
        ds,
        n_indices=args.eval_samples,
        per_stratum=per_stratum,
        seed=args.seed,
        out_path=out,
        stratify_fn=morphomnist_stratify(ds),
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
