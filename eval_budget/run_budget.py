"""Run the evaluation-budget experiment."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from spec import MPLCONFIG  # noqa: E402

os.environ["MPLCONFIGDIR"] = str(MPLCONFIG)
MPLCONFIG.mkdir(parents=True, exist_ok=True)

from compare_budget import compare_dataset  # noqa: E402
from ea_checkpoint import CURRENT, BudgetContext, install  # noqa: E402
from plot_objectives import plot_dataset  # noqa: E402
from score_budget import score_dataset  # noqa: E402
from snapshots import is_done, write_done  # noqa: E402
from spec import (  # noqa: E402
    RUN_IDS,
    SPECS,
    evolution_seed,
    load_config,
    load_manifest,
    manifest_sample_idx,
    run_dir,
)
from turtle_io import import_turtle, load_image  # noqa: E402


def _device(requested: str | None, cfg: dict[str, Any]) -> str:
    import torch

    choice = requested or str(cfg["runtime"].get("device", "auto"))
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def _imagenet_root(explicit: str | None) -> str | None:
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    env = os.environ.get("IMAGENET_ROOT", "").strip()
    return str(Path(env).expanduser().resolve()) if env else None


def evolve_job(payload: dict[str, Any]) -> tuple[int, int, str, str | None]:
    """One sample × seed. Safe to run again after a stop."""
    dataset = str(payload["dataset"])
    entry = payload["entry"]
    run_id = int(payload["run_id"])
    sample_idx = manifest_sample_idx(entry)
    y_true = int(entry["y_true"])
    cfg = load_config(dataset)
    budget = cfg["budget"]
    directory = run_dir(dataset, sample_idx, y_true, run_id)
    snapshot_gens = tuple(int(g) for g in budget["snapshot_gens"])
    max_generations = int(budget["max_generations"])
    if is_done(directory, max_generations=max_generations, snapshot_gens=snapshot_gens):
        return sample_idx, run_id, "skip", None

    previous = Path.cwd()
    try:
        turtle = import_turtle(dataset)
        install(turtle)
        _idx, y_loaded, image, _intended = load_image(
            dataset,
            turtle,
            entry,
            imagenet_root=payload.get("imagenet_root"),
        )
        if int(_idx) != sample_idx or int(y_loaded) != y_true:
            raise RuntimeError(
                f"Loaded sample {_idx} label {y_loaded}, expected {sample_idx} label {y_true}."
            )
        runtime = cfg["runtime"]
        spec = SPECS[dataset]
        job_cfg = dict(cfg["evolution"])
        job_cfg.update(
            {
                "population_size": int(budget["population_size"]),
                "max_generations": max_generations,
                "n_runs": 1,
                "run_id": run_id,
                "results_dir": str(directory) + os.sep,
                "device": payload["device"],
                "batch_size_eval": int(runtime["batch_size_eval"]),
                "amp": bool(runtime["amp"]),
                "model_ckpt": str(spec.model_ckpt),
                "turtle_delta_scale": float(runtime["turtle_delta_scale"]),
            }
        )
        directory.mkdir(parents=True, exist_ok=True)
        os.chdir(spec.graco_dir)
        ctx = BudgetContext(
            dataset=dataset,
            run_dir=directory,
            seed=evolution_seed(run_id),
            population_size=int(budget["population_size"]),
            max_generations=max_generations,
            snapshot_gens=snapshot_gens,
            label=y_true,
            image=image,
            turtle=turtle,
            device=str(payload["device"]),
            batch_size=int(runtime["batch_size_eval"]),
            amp=bool(runtime["amp"]),
            turtle_delta_scale=float(runtime["turtle_delta_scale"]),
            model_ckpt=str(spec.model_ckpt),
        )
        token = CURRENT.set(ctx)
        try:
            turtle.evolveImage(job_cfg, {"decideTurtle": turtle.decideTurtle}, image, y_true)
        finally:
            CURRENT.reset(token)
        write_done(
            directory,
            {
                "sample_idx": sample_idx,
                "y_true": y_true,
                "run_id": run_id,
                "seed": evolution_seed(run_id),
                "population_size": int(budget["population_size"]),
                "max_generations": max_generations,
                "snapshot_gens": list(snapshot_gens),
            },
        )
        return sample_idx, run_id, "ok", None
    except Exception as exc:
        return sample_idx, run_id, "error", str(exc)
    finally:
        os.chdir(previous)


def evolve_dataset(dataset: str, *, num_workers: int, device: str, imagenet_root: str | None) -> None:
    if dataset == "imagenet" and not imagenet_root:
        raise SystemExit("ImageNet evolution needs --imagenet-root or IMAGENET_ROOT.")
    spec = SPECS[dataset]
    if not spec.model_ckpt.is_file():
        raise FileNotFoundError(f"Missing classifier weights {spec.model_ckpt}")
    entries = load_manifest(dataset)
    jobs = []
    for entry in entries:
        for run_id in RUN_IDS:
            jobs.append(
                {
                    "dataset": dataset,
                    "entry": entry,
                    "run_id": int(run_id),
                    "device": device,
                    "imagenet_root": imagenet_root,
                }
            )
    print(
        f"[{dataset}] jobs={len(jobs)} population="
        f"{load_config(dataset)['budget']['population_size']} "
        f"generations={load_config(dataset)['budget']['max_generations']} "
        f"seeds={[evolution_seed(run_id) for run_id in RUN_IDS]} device={device}",
        flush=True,
    )
    n_ok = n_skip = n_fail = 0
    workers = max(1, int(num_workers))
    if workers == 1:
        for job in jobs:
            sample_idx, run_id, status, message = evolve_job(job)
            n_ok, n_skip, n_fail = _count(n_ok, n_skip, n_fail, sample_idx, run_id, status, message)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as pool:
            futures = [pool.submit(evolve_job, job) for job in jobs]
            for fut in as_completed(futures):
                sample_idx, run_id, status, message = fut.result()
                n_ok, n_skip, n_fail = _count(n_ok, n_skip, n_fail, sample_idx, run_id, status, message)
    print(f"[{dataset}] evolve finished ok={n_ok} skip={n_skip} failed={n_fail}", flush=True)
    if n_fail:
        raise SystemExit(f"{n_fail} evolution job(s) failed.")


def _count(
    n_ok: int,
    n_skip: int,
    n_fail: int,
    sample_idx: int,
    run_id: int,
    status: str,
    message: str | None,
) -> tuple[int, int, int]:
    if status == "ok":
        print(f"  ok sample={sample_idx} run={run_id} seed={evolution_seed(run_id)}", flush=True)
        return n_ok + 1, n_skip, n_fail
    if status == "skip":
        print(f"  skip sample={sample_idx} run={run_id}", flush=True)
        return n_ok, n_skip + 1, n_fail
    print(f"  ERROR sample={sample_idx} run={run_id}: {message}", flush=True)
    return n_ok, n_skip, n_fail + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--stage", choices=("evolve", "score", "compare", "all"), default="all")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel sample×run jobs for --stage evolve.")
    parser.add_argument("--device", default=None, help="auto, cpu, or cuda. Default: the dataset config.")
    parser.add_argument("--imagenet-root", default=None, help="ImageNet-100 root (train/ and val/).")
    args = parser.parse_args()

    imagenet_root = _imagenet_root(args.imagenet_root)
    if args.stage in {"evolve", "all"}:
        device = _device(args.device, load_config(args.dataset))
        evolve_dataset(
            args.dataset,
            num_workers=args.num_workers,
            device=device,
            imagenet_root=imagenet_root,
        )
    if args.stage in {"score", "all"}:
        score_dataset(args.dataset, imagenet_root=imagenet_root)
    if args.stage in {"compare", "all"}:
        compare_dataset(args.dataset)
        plot_dataset(args.dataset)


if __name__ == "__main__":
    main()
