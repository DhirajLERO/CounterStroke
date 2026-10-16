"""Run Graco Inception turtle evolution and merge Graco strategies into ``outputs/generation``."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_IMNET_DIR = Path(__file__).resolve().parent
_GRACO = _IMNET_DIR / "graco"


def _read_graco_paths(config_path: Path) -> tuple[Path, float]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    exp = cfg.get("__parallel_jobs_inception__", {}).get("experiment_name", "experiment_inception_turtle")
    results_root = cfg.get("paths", {}).get("results_root", "results")
    td = float(cfg.get("runtime", {}).get("turtle_delta_scale", 5.0))
    return (_GRACO / results_root / exp).resolve(), td


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--imagenet-root",
        type=Path,
        required=True,
        help="ImageNet root (required for turtle; pass the same path for --export-only for a consistent CLI).",
    )
    p.add_argument("--config", type=Path, default=_IMNET_DIR / "config.json")
    p.add_argument(
        "--manifest",
        type=Path,
        default=_IMNET_DIR / "data_subset" / "eval_manifest.json",
        help="Same eval_manifest.json as select_eval_samples (dataset_index, y_true).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--export-only",
        action="store_true",
        help="Skip turtle; merge existing Graco CSVs into outputs/generation (no re-evolution).",
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="After turtle, do not merge Graco strategies into generation.",
    )
    args = p.parse_args()

    if args.export_only and args.skip_export:
        raise SystemExit("Choose at most one of --export-only and --skip-export.")

    env = {**os.environ, "PYTHONPATH": str(_IMNET_DIR)}

    if not args.export_only:
        cmd = [
            sys.executable,
            str(_GRACO / "graco_inception_turtle.py"),
            "--config",
            str(args.config.resolve()),
            "--imagenet-root",
            str(args.imagenet_root.resolve()),
            "--manifest",
            str(args.manifest.resolve()),
            "--device",
            args.device,
        ]
        subprocess.check_call(cmd, cwd=str(_GRACO), env=env)

    if args.skip_export:
        print("generate_graco_inception finished (turtle only, --skip-export).")
        return

    graco_out, turtle_delta = _read_graco_paths(args.config)
    if not graco_out.is_dir():
        raise SystemExit(
            f"Missing Graco results dir {graco_out} (run turtle first or check "
            "paths.results_root / __parallel_jobs_inception__.experiment_name in config)."
        )

    base_gen = (_IMNET_DIR / "outputs" / "generation").resolve()
    out_graco = (_IMNET_DIR / "outputs" / "generation_graco").resolve()
    export_py = _GRACO / "export_graco_strategies.py"
    cmd2 = [
        sys.executable,
        str(export_py),
        "--base-gen-root",
        str(base_gen),
        "--graco-results-root",
        str(graco_out),
        "--out-gen-root",
        str(out_graco),
        "--turtle-delta-scale",
        str(turtle_delta),
    ]
    subprocess.check_call(cmd2, cwd=str(_IMNET_DIR), env=env)

    if out_graco.is_dir():
        for sdir in out_graco.glob("sample_*"):
            dest = base_gen / sdir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in sdir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, dest / f.name)

    print("generate_graco_inception finished")


if __name__ == "__main__":
    main()
