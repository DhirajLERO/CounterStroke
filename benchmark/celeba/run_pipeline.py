"""Resumable CelebA benchmark vs Graco pipeline (simple + complex graphs)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_CELEBA = Path(__file__).resolve().parent
_BENCHMARK = _CELEBA.parent
_GRACOIMAGE = _BENCHMARK.parent
sys.path.insert(0, str(_GRACOIMAGE))
sys.path.insert(0, str(_BENCHMARK / "shared"))

from common.pipeline_resume import mark_phase_done, should_run_phase  # noqa: E402
from scm_runner import train_scm, train_hvae_stages, train_gan_stages, evaluate_scm  # noqa: E402
from uncond_vae_paths import resolve_uncond_vae_ckpt  # noqa: E402
from paths import dataset_root  # noqa: E402
from subprocess_env import benchmark_subprocess_env, require_benchmark_deps  # noqa: E402
from eval_protocol import DEFAULT_EVAL_SAMPLES, GRACO_VARIANTS  # noqa: E402

MODELS = _CELEBA / "models"
PY = sys.executable


def run_phase(name: str, cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_CELEBA, env=benchmark_subprocess_env())
    mark_phase_done(MODELS, name)


def methods_for_graph(graph: str) -> list[str]:
    return ["vae", "hvae", "gan"]


def _uncond_ckpt_for_graph(
    args: argparse.Namespace,
    graph: str,
) -> Path | None:
    if graph == "simple":
        user = args.uncond_vae_ckpt_simple or args.uncond_vae_ckpt
    else:
        user = args.uncond_vae_ckpt_complex or args.uncond_vae_ckpt
    return resolve_uncond_vae_ckpt(
        user,
        dataset_root("celeba"),
        dataset="celeba",
        graph=graph,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", choices=["simple", "complex", "both"], default="both")
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-phase", type=str, default=None)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--skip-graco", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument(
        "--eval-samples",
        type=int,
        default=DEFAULT_EVAL_SAMPLES,
        help="Fixed test indices per graph for aligned SCM + Graco eval.",
    )
    p.add_argument(
        "--embeddings",
        type=str,
        default=None,
        choices=["vgg", "clfs", "vae", "lpips", "clip"],
        help="Embedding space for minimality (optional).",
    )
    p.add_argument(
        "--uncond-vae-ckpt",
        type=Path,
        default=None,
        help="Default uncond VAE path for all graphs (.ckpt file or directory).",
    )
    p.add_argument(
        "--uncond-vae-ckpt-simple",
        type=Path,
        default=None,
        help="Uncond VAE for CelebA simple graph (overrides --uncond-vae-ckpt for simple).",
    )
    p.add_argument(
        "--uncond-vae-ckpt-complex",
        type=Path,
        default=None,
        help="Uncond VAE for CelebA complex graph (overrides --uncond-vae-ckpt for complex).",
    )
    args = p.parse_args()
    require_benchmark_deps()

    graphs = ["simple", "complex"] if args.graph == "both" else [args.graph]

    def want(phase: str) -> bool:
        return should_run_phase(MODELS, phase, force=args.force, force_phase=args.force_phase, only_phase=args.only)

    MODELS.mkdir(parents=True, exist_ok=True)

    if want("verify_assets"):
        verify_cmd = [PY, str(_BENCHMARK / "shared" / "verify_assets.py"), "--dataset", "celeba"]
        if args.embeddings == "vae":
            verify_cmd.append("--embeddings")
            verify_cmd.append("vae")
            for g in graphs:
                ck = _uncond_ckpt_for_graph(args, g)
                if ck is None:
                    raise SystemExit(
                        f"When --embeddings vae, provide checkpoint for graph '{g}' "
                        f"(--uncond-vae-ckpt-{g} or --uncond-vae-ckpt) or use "
                        f"checkpoints/{g}/trained_uncond_vae/"
                    )
                verify_cmd.extend([f"--uncond-vae-ckpt-{g}", str(ck)])
        run_phase("verify_assets", verify_cmd)

    for graph in graphs:
        uncond_ckpt = _uncond_ckpt_for_graph(args, graph)
        if args.embeddings == "vae" and uncond_ckpt is None:
            raise SystemExit(
                f"When --embeddings vae, provide --uncond-vae-ckpt-{graph} or "
                f"--uncond-vae-ckpt, or place weights under checkpoints/{graph}/trained_uncond_vae/"
            )

        if want(f"build_eval_subset_{graph}"):
            run_phase(
                f"build_eval_subset_{graph}",
                [
                    PY,
                    str(_CELEBA / "build_eval_subset.py"),
                    "--graph",
                    graph,
                    "--eval-samples",
                    str(args.eval_samples),
                ],
            )

        cfg_dir = _CELEBA / "configs" / graph
        clf_cfg = cfg_dir / "classifier.json"
        eval_idx = _CELEBA / "data_subset" / f"eval_indices_{graph}.json"

        if not args.skip_train:
            for method in methods_for_graph(graph):
                cfg = cfg_dir / f"{method}.json"
                if not cfg.is_file():
                    continue
                phase = f"train_scm_{graph}_{method}"
                if want(phase):
                    if method == "hvae":
                        train_hvae_stages("celeba", cfg, graph=graph)
                    elif method == "gan":
                        train_gan_stages("celeba", cfg, graph=graph, classifier_config=clf_cfg)
                    else:
                        train_scm("celeba", cfg, graph=graph)
                    mark_phase_done(MODELS, phase)

        for method in methods_for_graph(graph):
            cfg = cfg_dir / f"{method}.json"
            if not cfg.is_file():
                continue
            phase = f"eval_scm_{graph}_{method}"
            if want(phase):
                out = _CELEBA / "outputs" / "scm" / graph / method
                evaluate_scm(
                    "celeba",
                    cfg,
                    clf_cfg,
                    out,
                    eval_indices=eval_idx,
                    graph=graph,
                    embeddings=args.embeddings,
                    uncond_vae_ckpt=uncond_ckpt,
                )
                mark_phase_done(MODELS, phase)

        if not args.skip_graco:
            if want(f"graco_evolve_{graph}"):
                run_phase(
                    f"graco_evolve_{graph}",
                    [
                        PY,
                        str(_CELEBA / "run_graco_evolve.py"),
                        "--graph",
                        graph,
                        "--max-samples",
                        str(args.eval_samples),
                    ],
                )
            for variant in GRACO_VARIANTS:
                phase = f"eval_graco_{graph}_{variant}"
                if want(phase):
                    graco_eval_cmd = [
                        PY,
                        str(_BENCHMARK / "shared" / "evaluate_graco.py"),
                        "--dataset",
                        "celeba",
                        "--graph",
                        graph,
                        "--variant",
                        variant,
                        "--scm-config",
                        str(cfg_dir / "vae.json"),
                        "--classifier-config",
                        str(clf_cfg),
                        "--cf-dir",
                        str(_CELEBA / "outputs" / "graco" / graph / variant),
                    ]
                    if args.embeddings:
                        graco_eval_cmd.extend(["--embeddings", args.embeddings])
                    if uncond_ckpt:
                        graco_eval_cmd.extend(["--uncond-vae-ckpt", str(uncond_ckpt)])
                    run_phase(phase, graco_eval_cmd)

    if want("aggregate_results"):
        run_phase(
            "aggregate_results",
            [
                PY,
                str(_BENCHMARK / "shared" / "aggregate_results.py"),
                "--outputs-root",
                str(_CELEBA / "outputs"),
                "--out-csv",
                str(_CELEBA / "outputs" / "summary.csv"),
            ],
        )


if __name__ == "__main__":
    main()
