"""Run per-sample metrics, CMMD/FID distribution metrics, and plots (ImageNet RGB)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_GRACOIMAGE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-root", type=Path, required=True)
    p.add_argument("--imagenet-root", type=Path, required=True)
    p.add_argument("--skip-distribution", action="store_true")
    args = p.parse_args()

    py = sys.executable
    gen = str(args.gen_root.resolve())
    eval_py = _GRACOIMAGE_ROOT / "common" / "eval_per_sample_imagenet.py"
    cmmd_py = _GRACOIMAGE_ROOT / "common" / "run_cmmd_imagenet.py"
    plot_py = _GRACOIMAGE_ROOT / "common" / "plot_methods.py"

    subprocess.check_call(
        [
            py,
            str(eval_py),
            "--gen-root",
            gen,
            "--csv-out",
            str(Path(gen) / "per_sample_metrics.csv"),
            "--batch-fid-csv",
            str(Path(gen) / "batch_fid.csv"),
            "--imagenet-root",
            str(args.imagenet_root.resolve()),
        ]
    )

    if not args.skip_distribution:
        subprocess.check_call(
            [
                py,
                str(cmmd_py),
                "--gen-root",
                gen,
                "--out-dir",
                gen,
                "--imagenet-root",
                str(args.imagenet_root.resolve()),
            ]
        )

    subprocess.check_call(
        [
            py,
            str(plot_py),
            "--gen-root",
            gen,
        ]
    )
    print("metrics_and_plots (ImageNet) done")


if __name__ == "__main__":
    main()
