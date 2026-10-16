"""Train / evaluate SCM methods via vendored deepscm scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from paths import DEEPSCM_DIR, dataset_root
from subprocess_env import benchmark_subprocess_env
from uncond_vae_paths import apply_uncond_vae_env


def _env(extra: dict | None = None) -> dict:
    return benchmark_subprocess_env(extra)


def _run(cmd: list[str], cwd: Path, log_path: Path | None = None, extra_env: dict | None = None) -> None:
    print("Running:", " ".join(cmd), "cwd=", cwd)
    run_env = _env(extra_env)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log:
            subprocess.check_call(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=run_env)
    else:
        subprocess.check_call(cmd, cwd=cwd, env=run_env)


def train_scm(
    dataset: str,
    config_path: Path,
    graph: str | None = None,
    *,
    classifier_config: Path | None = None,
) -> None:
    cwd = dataset_root(dataset)
    py = sys.executable
    train_py = DEEPSCM_DIR / "train.py"
    cfg_abs = config_path.resolve()
    log = cwd / "outputs" / "scm" / (graph or "default") / cfg_abs.stem / "train.log"
    extra_env: dict[str, str] = {}
    if classifier_config and classifier_config.is_file():
        extra_env["BENCHMARK_CLASSIFIER_CONFIG"] = str(classifier_config.resolve())
    _run([py, str(train_py), "-c", str(cfg_abs)], cwd=cwd, log_path=log, extra_env=extra_env or None)


def train_gan_stages(
    dataset: str,
    gan_config: Path,
    graph: str | None = None,
    *,
    classifier_config: Path | None = None,
) -> None:
    """Pretrain GAN (finetune=0) then cyclic fine-tune (finetune=1), matching upstream workflow."""
    cwd = dataset_root(dataset)
    data = json.loads(gan_config.read_text())
    rel = gan_config.relative_to(cwd)
    img_params = data["mechanism_models"]["image"]["params"]

    if img_params.get("finetune") != 1:
        train_scm(dataset, gan_config, graph=graph, classifier_config=classifier_config)
        return

    s1 = deepcopy(data)
    s1["mechanism_models"]["image"]["params"]["finetune"] = 0
    p1 = cwd / "configs" / f"_gan_pretrain_{rel.stem}.json"
    p1.write_text(json.dumps(s1, indent=4))
    train_scm(dataset, p1, graph=graph, classifier_config=classifier_config)

    ckpt_dir = Path(s1["checkpoint_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = cwd / ckpt_dir
    gan_ckpts = sorted(c for c in ckpt_dir.glob("image_gan-epoch*") if "finetuned" not in c.name)
    if not gan_ckpts:
        raise FileNotFoundError(f"No GAN pretrain checkpoint in {ckpt_dir}")
    pretrain_path = str(gan_ckpts[-1])

    s2 = deepcopy(data)
    s2["mechanism_models"]["image"]["params"]["finetune"] = 1
    s2["mechanism_models"]["image"]["params"]["pretrained_path"] = pretrain_path
    p2 = cwd / "configs" / f"_gan_finetune_{rel.stem}.json"
    p2.write_text(json.dumps(s2, indent=4))
    train_scm(dataset, p2, graph=graph, classifier_config=classifier_config)


def _hvae_needs_cf_finetune(dataset: str) -> bool:
    """Upstream runs CF fine-tune for CelebA HVAE only (MorphoMNIST/ADNI: ELBO pretrain)."""
    return dataset == "celeba"


def train_hvae_stages(dataset: str, hvae_config: Path, graph: str | None = None) -> None:
    """Pretrain ELBO; CelebA also CF fine-tunes with anti-causal classifiers."""
    cwd = dataset_root(dataset)
    data = json.loads(hvae_config.read_text())
    rel = hvae_config.relative_to(cwd)

    # Stage 1: pretrain
    s1 = deepcopy(data)
    s1["mechanism_models"]["image"]["params"]["cf_fine_tune"] = "False"
    s1["mechanism_models"]["image"]["params"]["evaluate_cf_model"] = "False"
    p1 = cwd / "configs" / f"_hvae_pretrain_{rel.stem}.json"
    p1.write_text(json.dumps(s1, indent=4))
    train_scm(dataset, p1, graph=graph)

    if not _hvae_needs_cf_finetune(dataset):
        print(f"HVAE for {dataset}: skipping CF fine-tune (upstream uses ELBO pretrain only).")
        return

    # Stage 2 (CelebA): CF fine-tune with elbo_constraint + classifier checkpoints.
    ckpt_dir = Path(s1["checkpoint_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = cwd / ckpt_dir
    image_ckpts = sorted(ckpt_dir.glob("*CondHVAE*epoch*")) or sorted(ckpt_dir.glob("*hvae*epoch*"))
    if not image_ckpts:
        raise FileNotFoundError(f"No HVAE checkpoint in {ckpt_dir}")
    pretrain_path = str(image_ckpts[-1])

    s2 = deepcopy(data)
    s2["mechanism_models"]["image"]["params"]["cf_fine_tune"] = "True"
    s2["mechanism_models"]["image"]["params"]["evaluate_cf_model"] = "False"
    s2["mechanism_models"]["image"]["params"]["checkpoint_path"] = pretrain_path
    p2 = cwd / "configs" / f"_hvae_finetune_{rel.stem}.json"
    p2.write_text(json.dumps(s2, indent=4))
    train_scm(dataset, p2, graph=graph)


def evaluate_scm(
    dataset: str,
    config_path: Path,
    classifier_config: Path,
    out_dir: Path,
    *,
    metrics: list[str] | None = None,
    eval_indices: Path | None = None,
    graph: str | None = None,
    embeddings: str | None = None,
    uncond_vae_ckpt: Path | None = None,
) -> None:
    cwd = dataset_root(dataset)
    py = sys.executable
    eval_py = DEEPSCM_DIR / "evaluate.py"
    mets = metrics or ["effectiveness", "fid", "minimality"]
    cmd = [
        py,
        str(eval_py),
        "-c",
        str(config_path.resolve()),
        "-clf",
        str(classifier_config.resolve()),
        "-m",
        *mets,
        "-qn",
        "0",
    ]
    if embeddings:
        cmd.extend(["--embeddings", embeddings])

    extra_env: dict[str, str] = {}
    if eval_indices and eval_indices.is_file():
        extra_env["BENCHMARK_EVAL_INDICES"] = str(eval_indices.resolve())
        interventions_path = eval_indices.parent / eval_indices.name.replace(
            "eval_indices", "eval_interventions"
        )
        if interventions_path.is_file():
            extra_env["BENCHMARK_EVAL_INTERVENTIONS"] = str(interventions_path.resolve())
    apply_uncond_vae_env(
        extra_env,
        embeddings=embeddings,
        uncond_vae_ckpt=uncond_vae_ckpt,
        celeba_graph=graph if dataset == "celeba" else None,
    )

    log = out_dir / "metrics.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(cmd, cwd=cwd, log_path=log, extra_env=extra_env)
