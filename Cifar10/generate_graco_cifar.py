"""Run Graco CIFAR turtle evolution + export three Graco strategies into generation folder."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_CIFAR_DIR = Path(__file__).resolve().parent
_GRACO_DIR = _CIFAR_DIR / "graco"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=_CIFAR_DIR / "data_subset" / "eval_manifest.json")
    p.add_argument("--config", type=Path, default=_CIFAR_DIR / "config.json")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--graco-results-subdir", type=str, default="experiment_cifar_turtle")
    p.add_argument("--base-from-generation", action="store_true", help="Use outputs/generation as base for export")
    args = p.parse_args()

    graco_py = _GRACO_DIR / "graco_cifar_turtle.py"
    if not graco_py.is_file():
        raise SystemExit(f"Missing {graco_py}")

    cmd = [
        sys.executable,
        str(graco_py),
        "--config",
        str(args.config.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--device",
        args.device,
    ]
    env = {**os.environ, "PYTHONPATH": str(_CIFAR_DIR)}
    subprocess.check_call(cmd, cwd=str(_GRACO_DIR), env=env)

    graco_out = _GRACO_DIR / "results" / args.graco_results_subdir
    if not graco_out.is_dir():
        graco_out = _CIFAR_DIR / "results" / args.graco_results_subdir

    base_gen = _CIFAR_DIR / "outputs" / "generation" if args.base_from_generation else _CIFAR_DIR / "outputs" / "generation"
    out_graco = _CIFAR_DIR / "outputs" / "generation_graco"
    export_py = _GRACO_DIR / "export_graco_strategies.py"
    cmd2 = [
        sys.executable,
        str(export_py),
        "--base-gen-root",
        str(base_gen.resolve()),
        "--graco-results-root",
        str(graco_out.resolve()),
        "--out-gen-root",
        str(out_graco.resolve()),
        "--cifar-root",
        str((_CIFAR_DIR / "data").resolve()),
    ]
    subprocess.check_call(cmd2, cwd=str(_CIFAR_DIR), env=env)

    if out_graco.is_dir():
        for sdir in out_graco.glob("sample_*"):
            dest = base_gen / sdir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in sdir.glob("*"):
                shutil.copy2(f, dest / f.name)

    print("generate_graco_cifar finished")


if __name__ == "__main__":
    main()
