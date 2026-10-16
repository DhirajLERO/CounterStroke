"""Random-grammar ablation on MNIST, CIFAR-10, and ImageNet-100."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import pickle
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

_GRACOIMAGE = Path(__file__).resolve().parents[1]
if str(_GRACOIMAGE) not in sys.path:
    sys.path.append(str(_GRACOIMAGE))

from engine import (  # noqa: E402
    DEFAULT_SCORE_BATCH,
    SEED,
    _bind_model,
    evaluation_budget,
    load_backend,
    sample_one_image,
)
from common.collect_multirun_classification import (  # noqa: E402
    distribution_rows,
    per_sample_rows,
    table_means,
)

_ORDER = ("mnist", "cifar10", "imagenet100")
_READOUTS = {
    "graco_base_o2": "Random_sparse",
    "graco_base_o3": "Random_prox",
    "graco50": "Random_50",
}
_SPECS = {
    "mnist": {
        "config": _GRACOIMAGE / "MNIST" / "config.json",
        "root": _GRACOIMAGE / "MNIST",
        "parallel_key": "__parallel_jobs_v8__",
        "eval": _GRACOIMAGE / "common" / "eval_per_sample_mnist.py",
        "cmmd": _GRACOIMAGE / "common" / "run_cmmd_mnist.py",
        "spatial": (28, 28, 1),
    },
    "cifar10": {
        "config": _GRACOIMAGE / "Cifar10" / "config.json",
        "root": _GRACOIMAGE / "Cifar10",
        "parallel_key": "__parallel_jobs_v8__",
        "eval": _GRACOIMAGE / "common" / "eval_per_sample_cifar10.py",
        "cmmd": _GRACOIMAGE / "common" / "run_cmmd_cifar10.py",
        "spatial": (32, 32, 3),
    },
    "imagenet100": {
        "config": _GRACOIMAGE / "Imagenet" / "config.json",
        "root": _GRACOIMAGE / "Imagenet",
        "parallel_key": "__parallel_jobs_inception__",
        "eval": _GRACOIMAGE / "common" / "eval_per_sample_imagenet.py",
        "cmmd": _GRACOIMAGE / "common" / "run_cmmd_imagenet.py",
        "spatial": (299, 299, 3),
    },
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_nhwc(dataset: str, image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if dataset == "mnist":
        return arr.reshape(1, 28, 28, 1)
    if dataset == "cifar10":
        return arr.reshape(1, 32, 32, 3)
    chw = arr.reshape(3, 299, 299)
    return np.transpose(chw, (1, 2, 0))[np.newaxis, ...].astype(np.float32)


def _blank(dataset: str) -> np.ndarray:
    shape = (1, *_SPECS[dataset]["spatial"])
    return np.full(shape, np.nan, dtype=np.float32)


def _load_model(turtle, dataset: str, cfg: dict, device: str):
    paths = cfg["paths"]
    root = _SPECS[dataset]["root"]
    if dataset == "imagenet100":
        ckpt = str((root / paths["model_ckpt"]).resolve())
        model = turtle.load_task_classifier(ckpt, device)
        return model
    ckpt = turtle.resolve_model_ckpt(str(paths["model_ckpt"]))
    if dataset == "mnist":
        model = turtle.MNISTNet().to(device)
    else:
        model = turtle.CifarResNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def _load_samples(turtle, dataset: str, cfg: dict, max_images: int, imagenet_root: Path | None):
    paths = cfg["paths"]
    root = _SPECS[dataset]["root"]
    graco = root / "graco"
    if dataset == "mnist":
        data_dir = str((graco / paths.get("data_dir", "../data")).resolve())
        ds = turtle.load_dataset(data_dir=data_dir, download=bool(paths.get("download", True)))
        rows = []
        for idx in range(min(max_images, len(ds))):
            img_t, y = ds[idx]
            rows.append((idx, int(y), np.ascontiguousarray(img_t.squeeze().numpy(), dtype=np.float32)))
        return rows
    if dataset == "cifar10":
        data_dir = str((graco / paths.get("data_dir", "../data")).resolve())
        ds = turtle.load_dataset(data_dir=data_dir, download=bool(paths.get("download", True)))
        rows = []
        for idx in range(min(max_images, len(ds))):
            img_t, y = ds[idx]
            rows.append((idx, int(y), np.ascontiguousarray(img_t.numpy(), dtype=np.float32)))
        return rows
    manifest = (root / paths["eval_manifest"]).resolve()
    entries = _read_json(manifest)
    image_root = imagenet_root or (root / paths["imagenet100_root"]).resolve()
    loaded = turtle.load_imagenet_val_manifest_samples(str(image_root), entries, max_samples=max_images)
    return [
        (int(did), int(yt), np.ascontiguousarray(arr, dtype=np.float32))
        for did, yt, arr in loaded
    ]


def _save_sample(gen_root: Path, dataset: str, sample_idx: int, label: int, result: dict, budget: int) -> None:
    folder = gen_root / f"sample_{sample_idx:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    picks = result["picks"]
    payload = {"orig_nhwc": _as_nhwc(dataset, result["factual"])}
    selected = {}
    for method, key in (
        ("graco_base_o2", "cf_graco_base_o2"),
        ("graco_base_o3", "cf_graco_base_o3"),
        ("graco50", "cf_graco50"),
    ):
        scored = picks[method]
        if scored is None:
            payload[key] = _blank(dataset)
            selected[method] = None
        else:
            payload[key] = _as_nhwc(dataset, scored.image)
            selected[method] = {
                "o1": scored.o1,
                "o2": scored.o2,
                "o3": scored.o3,
                "pred_class": scored.pred_class,
                "pred_prob": scored.pred_prob,
                "readout": _READOUTS[method],
            }
    np.savez_compressed(folder / "tensors.npz", **payload)
    meta = {
        "sample_idx": int(sample_idx),
        "y_true": int(label),
        "seed": SEED,
        "n_evaluations": int(budget),
        "n_drawn": int(result["n_drawn"]),
        "n_mapped": int(result["n_mapped"]),
        "n_flips": int(result["n_flips"]),
        "image_seed": _image_seed(sample_idx),
        "graco_selected_row": selected,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_GRACOIMAGE))


def _score_metrics(dataset: str, gen_root: Path, metrics_dir: Path, imagenet_root: Path | None) -> None:
    spec = _SPECS[dataset]
    eval_cmd = [
        sys.executable,
        str(spec["eval"]),
        "--gen-root",
        str(gen_root),
        "--csv-out",
        str(metrics_dir / "per_sample_metrics.csv"),
        "--batch-fid-csv",
        str(metrics_dir / "batch_fid.csv"),
    ]
    cmmd_cmd = [
        sys.executable,
        str(spec["cmmd"]),
        "--gen-root",
        str(gen_root),
        "--out-dir",
        str(metrics_dir),
    ]
    if dataset == "imagenet100":
        if imagenet_root is None:
            raise SystemExit("ImageNet metrics need --imagenet-root")
        eval_cmd.extend(["--imagenet-root", str(imagenet_root)])
        cmmd_cmd.extend(["--imagenet-root", str(imagenet_root)])
    _run(eval_cmd)
    _run(cmmd_cmd)


def _relabel(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        method = row.get("method")
        if method in _READOUTS:
            row["readout"] = _READOUTS[method]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    print(f"Wrote {path.resolve()}", flush=True)


_JSON_PATH = _GRACOIMAGE / "random_ablation" / "random_grammar_classification.json"
_TENSOR_KEYS = ("orig_nhwc", "cf_graco_base_o2", "cf_graco_base_o3", "cf_graco50")


def _image_seed(_sample_idx: int) -> int:
    """Same seed on every image, matching the reported run (seed 10)."""
    return SEED


def _seed_image(sample_idx: int) -> int:
    image_seed = _image_seed(sample_idx)
    random.seed(image_seed)
    np.random.seed(image_seed % (2**32))
    torch.manual_seed(image_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(image_seed)
    return image_seed


def _expected_indices(dataset: str, cfg: dict, max_images: int) -> list[int]:
    if dataset != "imagenet100":
        return list(range(max_images))
    manifest = (_SPECS[dataset]["root"] / cfg["paths"]["eval_manifest"]).resolve()
    entries = _read_json(manifest)
    if max_images > 0:
        entries = list(entries)[:max_images]
    indices = []
    for rec in entries:
        idx = int(rec.get("dataset_index", rec.get("sample_idx", -1)))
        if idx < 0:
            raise ValueError(f"Manifest entry missing dataset_index: {rec!r}")
        indices.append(idx)
    return indices


def _finished_meta(gen_root: Path, sample_idx: int, budget: int) -> dict | None:
    folder = gen_root / f"sample_{sample_idx:05d}"
    meta_path = folder / "meta.json"
    npz_path = folder / "tensors.npz"
    if not meta_path.is_file() or not npz_path.is_file():
        return None
    try:
        meta = _read_json(meta_path)
    except json.JSONDecodeError:
        return None
    if int(meta.get("seed", -1)) != SEED:
        return None
    if int(meta.get("n_evaluations", -1)) != int(budget):
        return None
    if int(meta.get("n_drawn", -1)) != int(budget):
        return None
    if int(meta.get("sample_idx", -1)) != int(sample_idx):
        return None
    try:
        with np.load(npz_path) as data:
            keys = set(data.files)
    except Exception:
        return None
    if not set(_TENSOR_KEYS) <= keys:
        return None
    progress = folder / "progress.pkl"
    if progress.is_file():
        progress.unlink()
    return meta


def _log_from_meta(meta: dict) -> dict:
    return {
        "sample_idx": int(meta["sample_idx"]),
        "y_true": int(meta["y_true"]),
        "n_flips": int(meta["n_flips"]),
        "n_mapped": int(meta["n_mapped"]),
        "n_drawn": int(meta["n_drawn"]),
    }


def _dataset_recorded(payload: dict, dataset: str, budget: int, n_images: int) -> bool:
    block = payload.get("datasets", {}).get(dataset)
    if not isinstance(block, dict):
        return False
    recorded = block.get("budget", {})
    means = block.get("table_means")
    per_sample = block.get("per_sample")
    return (
        int(recorded.get("seed", -1)) == SEED
        and int(recorded.get("n_evaluations_per_image", -1)) == int(budget)
        and int(recorded.get("n_images", -1)) == int(n_images)
        and isinstance(means, list)
        and len(means) == 3
        and isinstance(per_sample, list)
        and len(per_sample) > 0
    )


def _load_progress(folder: Path, sample_idx: int, budget: int) -> dict | None:
    path = folder / "progress.pkl"
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            blob = pickle.load(handle)
    except Exception as exc:
        print(f"  ignoring unreadable progress for sample_{sample_idx:05d} ({exc})", flush=True)
        return None
    if int(blob.get("seed", -1)) != SEED or int(blob.get("budget", -1)) != int(budget):
        return None
    if int(blob.get("sample_idx", -1)) != int(sample_idx):
        return None
    drawn = int(blob.get("n_drawn", 0))
    if drawn <= 0 or drawn > int(budget) or not isinstance(blob.get("picks"), dict):
        return None
    if "rng_random" not in blob or "rng_numpy" not in blob:
        return None
    return blob


def _save_progress(folder: Path, sample_idx: int, budget: int, state: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    blob = {
        "budget": int(budget),
        "seed": SEED,
        "sample_idx": int(sample_idx),
        "n_drawn": int(state["n_drawn"]),
        "n_mapped": int(state["n_mapped"]),
        "n_flips": int(state["n_flips"]),
        "picks": state["picks"],
        "history": list(state.get("history") or []),
        "best_obj": list(state.get("best_obj") or []),
        "next_gen": int(state.get("next_gen", 0)),
        "rng_random": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path = folder / "progress.pkl"
    tmp = folder / "progress.pkl.tmp"
    with tmp.open("wb") as handle:
        pickle.dump(blob, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _restore_rng(blob: dict) -> None:
    random.setstate(blob["rng_random"])
    np.random.set_state(blob["rng_numpy"])
    torch_state = blob["rng_torch"]
    if hasattr(torch_state, "cpu"):
        torch_state = torch_state.cpu()
    torch.set_rng_state(torch_state)
    cuda_state = blob.get("rng_cuda")
    if cuda_state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as exc:
            print(f"  could not restore CUDA RNG ({exc}); grammar draws still resume", flush=True)


def _fresh_payload() -> dict[str, Any]:
    return {
        "note": (
            "Random programs from the same grammar, with no selection, crossover, or mutation. "
            "Each image draws population_size * max_generations programs. "
            "Random_sparse, Random_prox, and Random_50 use the same three pick rules on that whole set."
        ),
        "seed": SEED,
        "methods": dict(_READOUTS),
        "datasets": {},
    }


def _load_payload_from(json_path: Path) -> dict[str, Any]:
    payload = _fresh_payload()
    if not json_path.is_file():
        return payload
    saved = _read_json(json_path)
    datasets = saved.get("datasets", {})
    if isinstance(datasets, dict):
        payload["datasets"] = datasets
    names = ", ".join(payload["datasets"]) or "none"
    print(f"Loaded {json_path.name} with finished datasets: {names}", flush=True)
    return payload


_GENERATION_COLUMNS = (
    "gen",
    "n_drawn",
    "min_o1",
    "min_o2",
    "min_o3",
    "pick_sparse_o1",
    "pick_sparse_o2",
    "pick_sparse_o3",
    "pick_prox_o1",
    "pick_prox_o2",
    "pick_prox_o3",
    "pick_50_o1",
    "pick_50_o2",
    "pick_50_o3",
)


def _write_generation_csv(folder: Path, history: list[dict]) -> None:
    """Same columns the evolution plot reads, plus the three pick objectives."""
    path = folder / "objectives_by_generation.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_GENERATION_COLUMNS)
        writer.writeheader()
        for row in history:
            out = {}
            for column in _GENERATION_COLUMNS:
                value = row.get(column)
                if value is None or (isinstance(value, float) and math.isinf(value)):
                    out[column] = ""
                else:
                    out[column] = value
            writer.writerow(out)


def _dataset_score_batch(dataset: str, args) -> int:
    chosen = {
        "mnist": args.mnist_score_batch,
        "cifar10": args.cifar_score_batch,
        "imagenet100": args.imagenet_score_batch,
    }[dataset]
    if chosen is not None:
        return int(chosen)
    if args.score_batch is not None:
        return int(args.score_batch)
    return int(DEFAULT_SCORE_BATCH[dataset])


def _dataset_workers(dataset: str, args) -> int:
    chosen = {
        "mnist": args.mnist_workers,
        "cifar10": args.cifar_workers,
        "imagenet100": args.imagenet_workers,
    }[dataset]
    return max(1, int(chosen))


def _process_one_image(
    turtle,
    export,
    dataset: str,
    sample_idx: int,
    label: int,
    factual: np.ndarray,
    evo: dict,
    budget: int,
    score_batch: int,
    scale: float,
    population_size: int,
    gen_root: Path,
    index: int,
    n_samples: int,
) -> dict:
    folder = gen_root / f"sample_{sample_idx:05d}"
    progress = _load_progress(folder, sample_idx, budget)
    resume = None
    if progress is not None:
        try:
            _restore_rng(progress)
        except Exception as exc:
            print(
                f"  could not resume sample_{sample_idx:05d} ({exc}); starting it again",
                flush=True,
            )
            stale = folder / "progress.pkl"
            if stale.is_file():
                stale.unlink()
            progress = None
        else:
            resume = progress
    if resume is None:
        _seed_image(sample_idx)
        print(
            f"{dataset} image {index}/{n_samples} idx={sample_idx} label={label}",
            flush=True,
        )
    else:
        print(
            f"{dataset} image {index}/{n_samples} idx={sample_idx} label={label} "
            f"resuming at {progress['n_drawn']}/{budget}",
            flush=True,
        )

    def _checkpoint(state: dict, folder: Path = folder, sample_idx: int = sample_idx) -> None:
        _save_progress(folder, sample_idx, budget, state)

    result = sample_one_image(
        turtle,
        export,
        dataset,
        factual,
        int(label),
        evo,
        budget,
        score_batch,
        scale,
        int(population_size),
        resume=resume,
        on_batch=_checkpoint,
    )
    result["factual"] = factual
    _save_sample(gen_root, dataset, sample_idx, label, result, budget)
    _write_generation_csv(folder, list(result.get("history") or []))
    progress_path = folder / "progress.pkl"
    if progress_path.is_file():
        progress_path.unlink()
    print(f"  saved sample_{sample_idx:05d} flips={result['n_flips']}", flush=True)
    return {
        "sample_idx": int(sample_idx),
        "y_true": int(label),
        "n_flips": int(result["n_flips"]),
        "n_mapped": int(result["n_mapped"]),
        "n_drawn": int(result["n_drawn"]),
    }


def _worker_one_image(job: dict) -> dict:
    """One image in its own process. Loads the model itself so workers can share a GPU."""
    dataset = job["dataset"]
    turtle, export = load_backend(dataset)
    model = _load_model(turtle, dataset, job["cfg"], job["device"])
    _bind_model(model, int(job["score_batch"]), job["device"], bool(job["amp"]))
    return _process_one_image(
        turtle,
        export,
        dataset,
        int(job["sample_idx"]),
        int(job["label"]),
        np.asarray(job["factual"], dtype=np.float32),
        job["evo"],
        int(job["budget"]),
        int(job["score_batch"]),
        float(job["scale"]),
        int(job["population_size"]),
        Path(job["gen_root"]),
        index=int(job["index"]),
        n_samples=int(job["n_samples"]),
    )


def _run_dataset(dataset: str, args, payload: dict, json_path: Path, results_root: Path) -> None:
    spec = _SPECS[dataset]
    cfg = _read_json(spec["config"])
    evo = cfg["evolution"]
    runtime = cfg.get("runtime", {})
    parallel = cfg.get(spec["parallel_key"], {})
    max_images = int(parallel.get("max_indices", 100))
    population = int(evo["population_size"])
    generations = int(args.max_generations) if args.max_generations is not None else int(evo["max_generations"])
    budget = evaluation_budget(population, generations)
    score_batch = _dataset_score_batch(dataset, args)
    num_workers = _dataset_workers(dataset, args)
    device = _device(args.device or runtime.get("device", "auto"))
    amp = bool(runtime.get("amp", False))
    scale = float(runtime.get("turtle_delta_scale", 5.0))
    imagenet_root = Path(args.imagenet_root).resolve() if args.imagenet_root else None
    out_root = results_root / dataset
    gen_root = out_root / "generation"
    gen_root.mkdir(parents=True, exist_ok=True)
    indices = _expected_indices(dataset, cfg, max_images)
    done = {
        sample_idx: meta
        for sample_idx in indices
        if (meta := _finished_meta(gen_root, sample_idx, budget)) is not None
    }
    print(
        f"=== {dataset}: {len(done)}/{len(indices)} images already saved, "
        f"{budget} programs/image, score batch {score_batch}, workers {num_workers}, device {device} ===",
        flush=True,
    )
    if len(done) == len(indices) and _dataset_recorded(payload, dataset, budget, len(indices)):
        print(f"{dataset} already finished, including metrics; skipping", flush=True)
        return

    image_logs = []
    if len(done) < len(indices):
        turtle, export = load_backend(dataset)
        samples = _load_samples(turtle, dataset, cfg, max_images, imagenet_root)
        pending = [
            (index, sample_idx, label, factual)
            for index, (sample_idx, label, factual) in enumerate(samples, start=1)
            if sample_idx not in done
        ]
        for index, (sample_idx, _label, _factual) in enumerate(samples, start=1):
            if sample_idx in done:
                print(
                    f"{dataset} image {index}/{len(samples)} idx={sample_idx} already saved",
                    flush=True,
                )
        n_pool = min(num_workers, len(pending))
        finished: dict[int, dict] = {}
        if n_pool <= 1:
            model = _load_model(turtle, dataset, cfg, device)
            _bind_model(model, score_batch, device, amp)
            for index, sample_idx, label, factual in pending:
                finished[sample_idx] = _process_one_image(
                    turtle,
                    export,
                    dataset,
                    sample_idx,
                    label,
                    factual,
                    evo,
                    budget,
                    score_batch,
                    scale,
                    population,
                    gen_root,
                    index,
                    len(samples),
                )
        else:
            jobs = [
                {
                    "dataset": dataset,
                    "cfg": cfg,
                    "sample_idx": int(sample_idx),
                    "label": int(label),
                    "factual": np.ascontiguousarray(factual, dtype=np.float32),
                    "evo": evo,
                    "budget": int(budget),
                    "score_batch": int(score_batch),
                    "scale": float(scale),
                    "population_size": int(population),
                    "device": device,
                    "amp": bool(amp),
                    "gen_root": str(gen_root),
                    "index": int(index),
                    "n_samples": len(samples),
                }
                for index, sample_idx, label, factual in pending
            ]
            print(f"{dataset}: starting {n_pool} workers for {len(jobs)} images", flush=True)
            ctx = mp.get_context("spawn")
            errors: list[BaseException] = []
            with ProcessPoolExecutor(max_workers=n_pool, mp_context=ctx) as pool:
                futures = [pool.submit(_worker_one_image, job) for job in jobs]
                for fut in as_completed(futures):
                    try:
                        row = fut.result()
                    except BaseException as exc:
                        errors.append(exc)
                        print(f"  ERROR {exc}", flush=True)
                        continue
                    finished[int(row["sample_idx"])] = row
            if errors:
                raise SystemExit(
                    f"{dataset}: {len(errors)} image(s) failed. Rerun the same command to continue."
                )
        image_logs = []
        for sample_idx, _label, _factual in samples:
            if sample_idx in done:
                image_logs.append(_log_from_meta(done[sample_idx]))
            else:
                image_logs.append(finished[int(sample_idx)])
    else:
        print(f"{dataset} images already saved; computing metrics", flush=True)
        image_logs = [_log_from_meta(done[sample_idx]) for sample_idx in indices]

    log = {
        "dataset": dataset,
        "seed": SEED,
        "population_size": population,
        "max_generations": generations,
        "n_evaluations_per_image": budget,
        "n_images": len(image_logs),
        "n_flips": int(sum(row["n_flips"] for row in image_logs)),
        "images": image_logs,
    }
    (out_root / "sample_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Saved {dataset} picks under {gen_root}", flush=True)

    _score_metrics(dataset, gen_root, out_root, imagenet_root)
    rows = per_sample_rows(out_root / "per_sample_metrics.csv")
    means = table_means(rows)
    dist = distribution_rows(out_root)
    _relabel(rows)
    _relabel(means)
    _relabel(dist)
    payload["datasets"][dataset] = {
        "budget": {
            "seed": SEED,
            "population_size": population,
            "max_generations": generations,
            "n_evaluations_per_image": budget,
            "n_images": len(image_logs),
            "n_flips": log["n_flips"],
            "score_batch": score_batch,
            "num_workers": num_workers,
        },
        "per_sample": rows,
        "table_means": means,
        "distribution": dist,
    }
    _write_json(json_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imagenet-root", type=Path, default=None, help="ImageNet-100 root with train/ and val/.")
    parser.add_argument("--device", type=str, default=None, help="auto, cpu, or cuda. Default: the config, else auto.")
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Override every dataset's generation count. Budget is population_size times this.",
    )
    parser.add_argument(
        "--score-batch",
        type=int,
        default=None,
        help="One GPU batch size for every dataset. Per-dataset flags override this.",
    )
    parser.add_argument("--mnist-score-batch", type=int, default=None, help="MNIST programs per GPU batch. Default 2048.")
    parser.add_argument("--cifar-score-batch", type=int, default=None, help="CIFAR-10 programs per GPU batch. Default 512.")
    parser.add_argument("--imagenet-score-batch", type=int, default=None, help="ImageNet-100 programs per GPU batch. Default 64.")
    parser.add_argument("--mnist-workers", type=int, default=1, help="MNIST images to run at once.")
    parser.add_argument("--cifar-workers", type=int, default=1, help="CIFAR-10 images to run at once.")
    parser.add_argument("--imagenet-workers", type=int, default=1, help="ImageNet-100 images to run at once.")
    args = parser.parse_args()

    if args.max_generations is None:
        results_root = _GRACOIMAGE / "random_ablation" / "results"
        json_path = _JSON_PATH
    else:
        tag = f"g{int(args.max_generations)}"
        results_root = _GRACOIMAGE / "random_ablation" / f"results_{tag}"
        json_path = _GRACOIMAGE / "random_ablation" / f"random_grammar_classification_{tag}.json"
    payload = _load_payload_from(json_path)
    for dataset in _ORDER:
        _run_dataset(dataset, args, payload, json_path, results_root)
    print(f"Random ablation finished. Copy {json_path.name} back.", flush=True)


if __name__ == "__main__":
    main()
