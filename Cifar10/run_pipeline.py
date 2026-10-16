"""Resumable CIFAR-10 pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_CIFAR_DIR = Path(__file__).resolve().parent
_GRACOIMAGE_ROOT = _CIFAR_DIR.parent
sys.path.insert(0, str(_GRACOIMAGE_ROOT))

from common.pipeline_resume import mark_phase_done, should_run_phase

MODELS = _CIFAR_DIR / "models"
PY = sys.executable


def run_phase(name: str, cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    mark_phase_done(MODELS, name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-phase", type=str, default=None)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--skip-graco", action="store_true")
    p.add_argument("--cifar-root", type=Path, default=_CIFAR_DIR / "data")
    args = p.parse_args()

    def want(phase: str) -> bool:
        return should_run_phase(MODELS, phase, force=args.force, force_phase=args.force_phase, only_phase=args.only)

    MODELS.mkdir(parents=True, exist_ok=True)

    if want("train_classifier"):
        run_phase(
            "train_classifier",
            [PY, str(_CIFAR_DIR / "train_classifier.py"), "--cifar-root", str(args.cifar_root)],
        )

    if want("compute_confusion_csv"):
        run_phase(
            "compute_confusion_csv",
            [
                PY,
                str(_CIFAR_DIR / "compute_confusion_csv.py"),
                "--cifar-root",
                str(args.cifar_root),
            ],
        )

    if want("select_eval_samples"):
        run_phase(
            "select_eval_samples",
            [
                PY,
                str(_CIFAR_DIR / "select_eval_samples.py"),
                "--cifar-root",
                str(args.cifar_root),
            ],
        )

    if want("train_explainer_models"):
        run_phase(
            "train_explainer_models",
            [PY, str(_CIFAR_DIR / "train_explainer_models.py"), "--cifar-root", str(args.cifar_root)],
        )

    gen = _CIFAR_DIR / "outputs" / "generation"

    if want("generate_counterfactual_pt"):
        run_phase(
            "generate_counterfactual_pt",
            [
                PY,
                str(_CIFAR_DIR / "generate_counterfactual_pt.py"),
                "--cifar-root",
                str(args.cifar_root),
            ],
        )

    if want("generate_cfproto"):
        run_phase(
            "generate_cfproto",
            [PY, str(_CIFAR_DIR / "generate_cfproto.py"), "--cifar-root", str(args.cifar_root)],
        )

    if want("generate_cfrl"):
        run_phase(
            "generate_cfrl",
            [PY, str(_CIFAR_DIR / "generate_cfrl.py"), "--cifar-root", str(args.cifar_root)],
        )

    if want("generate_leapfactual"):
        run_phase(
            "generate_leapfactual",
            [
                PY,
                str(_CIFAR_DIR / "generate_leapfactual.py"),
                "--cifar-root",
                str(args.cifar_root),
            ],
        )

    if not args.skip_graco and want("generate_graco_cifar"):
        run_phase(
            "generate_graco_cifar",
            [PY, str(_CIFAR_DIR / "generate_graco_cifar.py"), "--base-from-generation"],
        )

    if want("metrics_and_plots"):
        run_phase(
            "metrics_and_plots",
            [
                PY,
                str(_CIFAR_DIR / "metrics_and_plots.py"),
                "--gen-root",
                str(gen.resolve()),
                "--cifar-root",
                str(args.cifar_root),
            ],
        )

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
