"""Resumable MorphoMNIST benchmark vs Graco pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_MORPHO = Path(__file__).resolve().parent
_BENCHMARK = _MORPHO.parent
_GRACOIMAGE = _BENCHMARK.parent
sys.path.insert(0, str(_GRACOIMAGE))
sys.path.insert(0, str(_BENCHMARK / "shared"))

from common.pipeline_resume import mark_phase_done, should_run_phase  # noqa: E402
from scm_runner import train_scm, train_hvae_stages, train_gan_stages, evaluate_scm  # noqa: E402
from uncond_vae_paths import resolve_uncond_vae_ckpt  # noqa: E402
from paths import dataset_root  # noqa: E402
from subprocess_env import benchmark_subprocess_env, require_benchmark_deps  # noqa: E402
from eval_protocol import DEFAULT_EVAL_SAMPLES, GRACO_VARIANTS  # noqa: E402

MODELS = _MORPHO / "models"
PY = sys.executable
METHODS = ["vae", "hvae", "gan"]


def run_phase(name: str, cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_MORPHO, env=benchmark_subprocess_env())
    mark_phase_done(MODELS, name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-phase", type=str, default=None)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--skip-graco", action="store_true")
    p.add_argument("--skip-train", action="store_true", help="Skip SCM training (eval only)")
    p.add_argument(
        "--eval-samples",
        type=int,
        default=DEFAULT_EVAL_SAMPLES,
        help="Number of fixed test indices for aligned SCM + Graco eval.",
    )
    p.add_argument(
        "--embeddings",
        type=str,
        default=None,
        choices=["vgg", "clfs", "vae", "lpips", "clip"],
        help="Embedding space for minimality (optional). Use vae for CLD-style minimality.",
    )
    p.add_argument(
        "--uncond-vae-ckpt",
        type=Path,
        default=None,
        help="Unconditional VAE .ckpt file or directory (required when --embeddings vae). "
        "Default: checkpoints/trained_uncond_vae/",
    )
    args = p.parse_args()
    require_benchmark_deps()

    def want(phase: str) -> bool:
        return should_run_phase(MODELS, phase, force=args.force, force_phase=args.force_phase, only_phase=args.only)

    MODELS.mkdir(parents=True, exist_ok=True)

    root = dataset_root("morphomnist")
    uncond_ckpt = resolve_uncond_vae_ckpt(
        args.uncond_vae_ckpt,
        root,
        dataset="morphomnist",
    )
    if args.embeddings == "vae" and uncond_ckpt is None:
        raise SystemExit(
            "When --embeddings vae, provide --uncond-vae-ckpt or place weights under "
            "checkpoints/trained_uncond_vae/"
        )

    verify_cmd = [PY, str(_BENCHMARK / "shared" / "verify_assets.py"), "--dataset", "morphomnist"]
    if args.embeddings == "vae":
        verify_cmd.extend(["--embeddings", "vae", "--uncond-vae-ckpt", str(uncond_ckpt)])

    if want("verify_assets"):
        run_phase("verify_assets", verify_cmd)

    if want("build_eval_subset"):
        run_phase(
            "build_eval_subset",
            [
                PY,
                str(_MORPHO / "build_eval_subset.py"),
                "--eval-samples",
                str(args.eval_samples),
            ],
        )

    clf_cfg = _MORPHO / "configs" / "classifier.json"
    eval_idx = _MORPHO / "data_subset" / "eval_indices.json"

    if not args.skip_train:
        for method in METHODS:
            cfg = _MORPHO / "configs" / f"{method}.json"
            if not cfg.is_file():
                continue
            phase = f"train_scm_{method}"
            if want(phase):
                if method == "hvae":
                    train_hvae_stages("morphomnist", cfg)
                elif method == "gan":
                    train_gan_stages("morphomnist", cfg, classifier_config=clf_cfg)
                else:
                    train_scm("morphomnist", cfg)
                mark_phase_done(MODELS, phase)

    for method in METHODS:
        cfg = _MORPHO / "configs" / f"{method}.json"
        if not cfg.is_file():
            continue
        phase = f"eval_scm_{method}"
        if want(phase):
            out = _MORPHO / "outputs" / "scm" / "default" / method
            evaluate_scm(
                "morphomnist",
                cfg,
                clf_cfg,
                out,
                eval_indices=eval_idx,
                embeddings=args.embeddings,
                uncond_vae_ckpt=uncond_ckpt,
            )
            mark_phase_done(MODELS, phase)

    if not args.skip_graco:
        if want("graco_evolve"):
            run_phase(
                "graco_evolve",
                [
                    PY,
                    str(_MORPHO / "run_graco_evolve.py"),
                    "--max-samples",
                    str(args.eval_samples),
                ],
            )
        if want("eval_graco"):
            for variant in GRACO_VARIANTS:
                graco_eval_cmd = [
                    PY,
                    str(_BENCHMARK / "shared" / "evaluate_graco.py"),
                    "--dataset",
                    "morphomnist",
                    "--variant",
                    variant,
                    "--scm-config",
                    str(_MORPHO / "configs" / "vae.json"),
                    "--classifier-config",
                    str(clf_cfg),
                    "--cf-dir",
                    str(_MORPHO / "outputs" / "graco" / variant),
                ]
                if args.embeddings:
                    graco_eval_cmd.extend(["--embeddings", args.embeddings])
                if uncond_ckpt:
                    graco_eval_cmd.extend(["--uncond-vae-ckpt", str(uncond_ckpt)])
                run_phase(f"eval_graco_{variant}", graco_eval_cmd)

    if want("aggregate_results"):
        run_phase(
            "aggregate_results",
            [
                PY,
                str(_BENCHMARK / "shared" / "aggregate_results.py"),
                "--outputs-root",
                str(_MORPHO / "outputs"),
                "--out-csv",
                str(_MORPHO / "outputs" / "summary.csv"),
            ],
        )


if __name__ == "__main__":
    main()
