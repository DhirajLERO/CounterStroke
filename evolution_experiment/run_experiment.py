"""Run Graco evolution from this folder using MNIST / CIFAR / ImageNet turtles."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from wrap_graco import (  # noqa: E402
    SPECS,
    build_job_cfg,
    config_num_workers,
    evolve_one_run,
    import_turtle,
    load_experiment_config,
    load_manifest_entries,
    load_samples,
    parallel_evolve_job,
    resolve_device,
    results_dir_for,
    sample_run_dir,
    select_manifest_entries,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="30-run (or 100-example) Graco evolution for mean-best-fitness plots."
    )
    p.add_argument("--dataset", choices=tuple(SPECS), required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config (default: evolution_experiment/configs/<dataset>_30runs.json).",
    )
    p.add_argument("--manifest", type=Path, default=None, help="Override eval_manifest.json")
    p.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Dataset index of the single example (default: first eval-manifest entry).",
    )
    p.add_argument("--n-runs", type=int, default=30, help="Runs per example with seeds 0 .. n-runs-1")
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="How many runs to evolve at once. Default: config "
        "__parallel_jobs_*.num_workers (1). Use 8 to run 8 seeds in parallel.",
    )
    p.add_argument(
        "--max-indices",
        type=int,
        default=1,
        help="How many examples to evolve. Use 100 with --n-runs 1 for the 100-example plot.",
    )
    p.add_argument("--device", type=str, default=None, help="auto|cpu|cuda")
    p.add_argument(
        "--imagenet-root",
        type=Path,
        default=None,
        help="ImageNet-100 root (train/ and val/). Required for --dataset imagenet if not in config.",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Where to write {idx}_label_{y}/run_*/ (default: evolution_experiment/results/<dataset>).",
    )
    p.add_argument(
        "--test-run",
        action="store_true",
        help="Tiny population/generations for a smoke test.",
    )
    p.add_argument(
        "--skip-plot",
        action="store_true",
        help="Do not plot after the runs finish.",
    )
    args = p.parse_args()

    spec = SPECS[args.dataset]
    cfg = load_experiment_config(spec, args.config)
    device = resolve_device(args.device, cfg)
    results_root = (args.results_root or results_dir_for(args.dataset)).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    n_runs = max(1, int(args.n_runs))
    max_indices = max(1, int(args.max_indices))
    if args.sample_idx is not None:
        max_indices = 1
    cfg_workers = config_num_workers(spec, cfg)
    num_workers = max(1, int(args.num_workers if args.num_workers is not None else cfg_workers))

    entries = load_manifest_entries(spec, args.manifest)
    selected = select_manifest_entries(
        entries, sample_idx=args.sample_idx, max_indices=max_indices
    )
    config_path = (args.config or spec.derived_config).resolve()

    turtle = import_turtle(args.dataset)
    samples = load_samples(
        spec,
        turtle,
        cfg,
        selected,
        imagenet_root=args.imagenet_root,
    )
    if not samples:
        raise SystemExit("No samples loaded.")

    jobs: list[dict[str, Any]] = []
    for sample_idx, y_true, image in samples:
        for run_id in range(n_runs):
            jobs.append(
                {
                    "dataset": args.dataset,
                    "config": str(config_path) if config_path.is_file() else None,
                    "image": image.copy(),
                    "sample_idx": int(sample_idx),
                    "y_true": int(y_true),
                    "run_id": int(run_id),
                    "seed": int(run_id),
                    "run_dir": str(sample_run_dir(results_root, sample_idx, y_true, run_id)),
                    "device": device,
                    "test_run": bool(args.test_run),
                }
            )
    num_workers = min(num_workers, max(1, len(jobs)))
    print(
        f"[{spec.name}] device={device} n_runs={n_runs} examples={len(selected)} "
        f"jobs={len(jobs)} num_workers={num_workers} results={results_root}"
    )

    n_ok = 0
    n_fail = 0
    if num_workers <= 1:
        for job in jobs:
            print(
                f"  sample={job['sample_idx']} label={job['y_true']} "
                f"run={job['run_id']} seed={job['seed']}"
            )
            try:
                job_cfg = build_job_cfg(
                    spec,
                    cfg,
                    results_dir=Path(job["run_dir"]),
                    device=device,
                    test_run=args.test_run,
                )
                trace = evolve_one_run(
                    spec,
                    turtle,
                    job_cfg,
                    image=job["image"],
                    label=int(job["y_true"]),
                    sample_idx=int(job["sample_idx"]),
                    run_id=int(job["run_id"]),
                    seed=int(job["seed"]),
                )
                print(f"    wrote {trace}")
                n_ok += 1
            except Exception as exc:
                n_fail += 1
                print(f"    ERROR sample={job['sample_idx']} run={job['run_id']}: {exc}")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
            futures = [pool.submit(parallel_evolve_job, job) for job in jobs]
            for fut in as_completed(futures):
                try:
                    sample_idx, y_true, run_id, status, message = fut.result()
                except Exception as exc:
                    n_fail += 1
                    print(f"  ERROR worker crashed: {exc}")
                    continue
                if status == "ok":
                    n_ok += 1
                    print(f"  ok sample={sample_idx} label={y_true} run={run_id} {message}")
                else:
                    n_fail += 1
                    print(f"  ERROR sample={sample_idx} run={run_id}: {message}")

    print(f"Finished {n_ok} ok / {n_fail} failed")
    if args.skip_plot or n_ok == 0:
        return

    from plot_evolution import plot_examples100, plot_runs30

    if len(samples) == 1 and n_runs > 1:
        plot_runs30(
            dataset=args.dataset,
            results_root=results_root,
            sample_idx=int(samples[0][0]),
        )
    elif len(samples) > 1:
        plot_examples100(dataset=args.dataset, results_root=results_root)
    else:
        plot_runs30(dataset=args.dataset, results_root=results_root)


if __name__ == "__main__":
    main()
