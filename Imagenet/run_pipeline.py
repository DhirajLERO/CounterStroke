"""Resumable ImageNet-100 pipeline (phase sentinels under ``Imagenet/models/``)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_IMNET_DIR = Path(__file__).resolve().parent
_ROOT = _IMNET_DIR.parent
sys.path.insert(0, str(_ROOT))

from common.pipeline_resume import mark_phase_done, should_run_phase

MODELS = _IMNET_DIR / "models"
PY = sys.executable


def run_phase(name: str, cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    mark_phase_done(MODELS, name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--imagenet-root",
        type=Path,
        required=True,
        help="ImageNet-100 root (train/val ImageFolder layout).",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-phase", type=str, default=None)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--skip-graco", action="store_true")
    args = p.parse_args()

    root = str(args.imagenet_root.resolve())

    def want(phase: str) -> bool:
        return should_run_phase(
            MODELS,
            phase,
            force=args.force,
            force_phase=args.force_phase,
            only_phase=args.only,
        )

    MODELS.mkdir(parents=True, exist_ok=True)

    if want("download_imagenet100"):
        run_phase(
            "download_imagenet100",
            [
                PY,
                str(_IMNET_DIR / "download_imagenet100.py"),
                "--out-root",
                root,
            ],
        )

    if want("train_classifier"):
        run_phase(
            "train_classifier",
            [
                PY,
                str(_IMNET_DIR / "train_classifier.py"),
                "--imagenet-root",
                root,
            ],
        )

    if want("compute_confusion_csv"):
        run_phase(
            "compute_confusion_csv",
            [
                PY,
                str(_IMNET_DIR / "compute_confusion_csv.py"),
                "--imagenet-root",
                root,
            ],
        )

    if want("select_eval_samples"):
        run_phase(
            "select_eval_samples",
            [
                PY,
                str(_IMNET_DIR / "select_eval_samples.py"),
                "--imagenet-root",
                root,
            ],
        )

    if want("train_explainer_models"):
        run_phase(
            "train_explainer_models",
            [
                PY,
                str(_IMNET_DIR / "train_explainer_models.py"),
                "--imagenet-root",
                root,
            ],
        )

    gen = _IMNET_DIR / "outputs" / "generation"
    gen_args = ["--imagenet-root", root]

    if want("generate_counterfactual_pt"):
        run_phase(
            "generate_counterfactual_pt",
            [PY, str(_IMNET_DIR / "generate_counterfactual_pt.py"), *gen_args],
        )

    if want("generate_cfproto"):
        run_phase(
            "generate_cfproto",
            [PY, str(_IMNET_DIR / "generate_cfproto.py"), *gen_args],
        )

    if want("generate_cfrl"):
        run_phase(
            "generate_cfrl",
            [PY, str(_IMNET_DIR / "generate_cfrl.py"), *gen_args],
        )

    if want("generate_leapfactual"):
        run_phase(
            "generate_leapfactual",
            [PY, str(_IMNET_DIR / "generate_leapfactual.py"), *gen_args],
        )

    if not args.skip_graco and want("generate_graco_inception"):
        run_phase(
            "generate_graco_inception",
            [PY, str(_IMNET_DIR / "generate_graco_inception.py"), "--imagenet-root", root],
        )

    if want("metrics_and_plots"):
        run_phase(
            "metrics_and_plots",
            [
                PY,
                str(_IMNET_DIR / "metrics_and_plots.py"),
                "--gen-root",
                str(gen.resolve()),
                "--imagenet-root",
                root,
            ],
        )

    print("ImageNet pipeline finished.")


if __name__ == "__main__":
    main()
