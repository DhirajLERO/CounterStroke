"""Run per-sample metrics, CMMD/FID distribution metrics, and Graco-aware plots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_GRACOIMAGE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-root", type=Path, required=True)
    p.add_argument("--cifar-root", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--pt-weights", type=Path, default=Path(__file__).resolve().parent / "models" / "cifar_resnet.pt")
    p.add_argument("--skip-distribution", action="store_true")
    args = p.parse_args()

    py = sys.executable
    gen = str(args.gen_root.resolve())
    eval_py = _GRACOIMAGE_ROOT / "common" / "eval_per_sample_cifar10.py"
    cmmd_py = _GRACOIMAGE_ROOT / "common" / "run_cmmd_cifar10.py"
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
            "--cifar-root",
            str(args.cifar_root.resolve()),
            "--pt-weights",
            str(args.pt_weights.resolve()),
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
                "--cifar-root",
                str(args.cifar_root.resolve()),
                "--pt-weights",
                str(args.pt_weights.resolve()),
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
    print("metrics_and_plots done")


if __name__ == "__main__":
    main()
