"""Import existing Graco turtles without editing them; force seeds and save traces."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
GRACOIMAGE_ROOT = EXPERIMENT_DIR.parent

_mpl_dir = EXPERIMENT_DIR / ".mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

OBJECTIVE_NAMES = ("o1_validity", "o2_sparsity", "o3_proximity")
OBJECTIVE_LABELS = (
    "O1 (validity)",
    "O2 (sparsity)",
    "O3 (proximity)",
)

_LOADED_TURTLES: dict[str, ModuleType] = {}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_dir: Path
    graco_dir: Path
    turtle_py: Path
    source_config: Path
    derived_config: Path
    manifest: Path
    default_ckpt: Path
    parallel_key: str
    data_dir: Path


SPECS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        dataset_dir=GRACOIMAGE_ROOT / "MNIST",
        graco_dir=GRACOIMAGE_ROOT / "MNIST" / "graco",
        turtle_py=GRACOIMAGE_ROOT / "MNIST" / "graco" / "graco_v8_turtle.py",
        source_config=GRACOIMAGE_ROOT / "MNIST" / "config.json",
        derived_config=EXPERIMENT_DIR / "configs" / "mnist_30runs.json",
        manifest=GRACOIMAGE_ROOT / "MNIST" / "data_subset" / "eval_manifest.json",
        default_ckpt=GRACOIMAGE_ROOT / "MNIST" / "models" / "mnist_cnn.pt",
        parallel_key="__parallel_jobs_v8__",
        data_dir=GRACOIMAGE_ROOT / "MNIST" / "data",
    ),
    "cifar": DatasetSpec(
        name="cifar",
        dataset_dir=GRACOIMAGE_ROOT / "Cifar10",
        graco_dir=GRACOIMAGE_ROOT / "Cifar10" / "graco",
        turtle_py=GRACOIMAGE_ROOT / "Cifar10" / "graco" / "graco_cifar_turtle.py",
        source_config=GRACOIMAGE_ROOT / "Cifar10" / "config.json",
        derived_config=EXPERIMENT_DIR / "configs" / "cifar_30runs.json",
        manifest=GRACOIMAGE_ROOT / "Cifar10" / "data_subset" / "eval_manifest.json",
        default_ckpt=GRACOIMAGE_ROOT / "Cifar10" / "models" / "cifar_resnet.pt",
        parallel_key="__parallel_jobs_v8__",
        data_dir=GRACOIMAGE_ROOT / "Cifar10" / "data",
    ),
    "imagenet": DatasetSpec(
        name="imagenet",
        dataset_dir=GRACOIMAGE_ROOT / "Imagenet",
        graco_dir=GRACOIMAGE_ROOT / "Imagenet" / "graco",
        turtle_py=GRACOIMAGE_ROOT / "Imagenet" / "graco" / "graco_inception_turtle.py",
        source_config=GRACOIMAGE_ROOT / "Imagenet" / "config.json",
        derived_config=EXPERIMENT_DIR / "configs" / "imagenet_30runs.json",
        manifest=GRACOIMAGE_ROOT / "Imagenet" / "data_subset" / "eval_manifest.json",
        default_ckpt=GRACOIMAGE_ROOT / "Imagenet" / "models" / "imagenet100_inception.pt",
        parallel_key="__parallel_jobs_inception__",
        data_dir=GRACOIMAGE_ROOT / "Imagenet" / "data" / "imagenet100",
    ),
}


def results_dir_for(dataset: str) -> Path:
    return EXPERIMENT_DIR / "results" / dataset


def plots_dir_for(dataset: str) -> Path:
    return EXPERIMENT_DIR / "plots" / dataset


def data_dir_for(dataset: str) -> Path:
    return EXPERIMENT_DIR / "data" / dataset


def sample_run_dir(results_root: Path, sample_idx: int, y_true: int, run_id: int) -> Path:
    return results_root / f"{int(sample_idx)}_label_{int(y_true)}" / f"run_{int(run_id)}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment_config(spec: DatasetSpec, config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or spec.derived_config
    if path.is_file():
        return load_json(path)
    if spec.source_config.is_file():
        return load_json(spec.source_config)
    raise FileNotFoundError(f"No config for {spec.name}: tried {path} and {spec.source_config}")


def load_manifest_entries(spec: DatasetSpec, manifest_path: Path | None = None) -> list[dict[str, Any]]:
    path = manifest_path or spec.manifest
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Manifest must be a JSON list: {path}")
    return raw


def _manifest_index(entry: dict[str, Any]) -> int:
    return int(entry.get("dataset_index", entry.get("sample_idx", -1)))


def select_manifest_entries(
    entries: list[dict[str, Any]],
    *,
    sample_idx: int | None,
    max_indices: int,
) -> list[dict[str, Any]]:
    if sample_idx is not None:
        match = [e for e in entries if _manifest_index(e) == int(sample_idx)]
        if match:
            return match[:1]
        return [{"sample_idx": int(sample_idx), "dataset_index": int(sample_idx)}]
    if not entries:
        return [{"sample_idx": i, "dataset_index": i} for i in range(max(1, int(max_indices)))]
    return entries[: max(1, int(max_indices))]


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextmanager
def _locked_seed(seed: int) -> Iterator[None]:
    """Force np/random/torch seeds; swallow turtle's internal ``10 + i`` reseeds."""
    import torch

    rng_seed = int(seed)
    orig_np_seed = np.random.seed
    orig_py_seed = random.seed
    orig_torch_seed = torch.manual_seed

    def _np_seed(*_args: Any, **_kwargs: Any) -> None:
        orig_np_seed(rng_seed)

    def _py_seed(*_args: Any, **_kwargs: Any) -> None:
        orig_py_seed(rng_seed)

    def _torch_seed(*_args: Any, **_kwargs: Any) -> Any:
        return orig_torch_seed(rng_seed)

    orig_np_seed(rng_seed)
    orig_py_seed(rng_seed)
    orig_torch_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)

    np.random.seed = _np_seed  # type: ignore[assignment]
    random.seed = _py_seed  # type: ignore[assignment]
    torch.manual_seed = _torch_seed  # type: ignore[assignment]
    try:
        yield
    finally:
        np.random.seed = orig_np_seed  # type: ignore[assignment]
        random.seed = orig_py_seed  # type: ignore[assignment]
        torch.manual_seed = orig_torch_seed  # type: ignore[assignment]


def _min_per_obj(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.astype(np.float64, copy=False)
    return np.nanmin(arr, axis=0)


def _avg_per_obj(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.astype(np.float64, copy=False)
    return np.nanmean(arr, axis=0)


def _stack_obj_stat(logbook: Any, field: str) -> np.ndarray:
    rows = logbook.select(field)
    stacked = []
    for row in rows:
        arr = np.asarray(row, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            arr = np.pad(arr, (0, 3 - arr.size), constant_values=np.nan)
        stacked.append(arr[:3])
    return np.stack(stacked, axis=0) if stacked else np.zeros((0, 3), dtype=np.float64)


def write_evolution_trace(
    logbook: Any,
    out_path: Path,
    *,
    seed: int,
    sample_idx: int,
    y_true: int,
    run_id: int,
) -> None:
    gen = np.asarray(logbook.select("gen"), dtype=np.int64)
    best = _stack_obj_stat(logbook, "min_obj")
    mean = _stack_obj_stat(logbook, "avg_obj")
    if best.shape[0] != gen.shape[0]:
        raise RuntimeError(
            f"Trace length mismatch: gen={gen.shape[0]} best={best.shape[0]} at {out_path}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        gen=gen,
        best=best.astype(np.float64),
        mean=mean.astype(np.float64),
        seed=np.int64(seed),
        sample_idx=np.int64(sample_idx),
        y_true=np.int64(y_true),
        run_id=np.int64(run_id),
        objective_names=np.array(OBJECTIVE_NAMES, dtype=object),
    )


def import_turtle(dataset: str) -> ModuleType:
    """Import one dataset turtle. Call at most one dataset per process."""
    if dataset in _LOADED_TURTLES:
        return _LOADED_TURTLES[dataset]
    if _LOADED_TURTLES:
        loaded = next(iter(_LOADED_TURTLES))
        raise RuntimeError(
            f"Turtle for {loaded!r} is already imported in this process; "
            "run each dataset in a separate process."
        )
    spec = SPECS[dataset]
    if not spec.turtle_py.is_file():
        raise FileNotFoundError(spec.turtle_py)

    for p in (str(spec.graco_dir), str(spec.dataset_dir)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    mod_name = f"_evo_turtle_{dataset}"
    with _working_directory(spec.graco_dir):
        module_spec = importlib.util.spec_from_file_location(mod_name, spec.turtle_py)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Cannot load {spec.turtle_py}")
        mod = importlib.util.module_from_spec(module_spec)
        sys.modules[mod_name] = mod
        module_spec.loader.exec_module(mod)

    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    if hasattr(mod, "save_detailed_report_pdf"):
        mod.save_detailed_report_pdf = _noop  # type: ignore[assignment]
    if hasattr(mod, "save_final_population_pickle"):
        mod.save_final_population_pickle = _noop  # type: ignore[assignment]

    _LOADED_TURTLES[dataset] = mod
    return mod


def resolve_model_ckpt(spec: DatasetSpec, cfg: dict[str, Any]) -> str:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    raw = str(paths.get("model_ckpt", spec.default_ckpt))
    p = Path(raw)
    if p.is_file():
        return str(p.resolve())
    if spec.default_ckpt.is_file():
        return str(spec.default_ckpt.resolve())
    candidates = [
        (spec.dataset_dir / raw).resolve(),
        (spec.graco_dir / raw).resolve(),
        (GRACOIMAGE_ROOT / raw).resolve(),
        spec.default_ckpt.resolve(),
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return str(spec.default_ckpt.resolve())


def config_num_workers(spec: DatasetSpec, cfg: dict[str, Any]) -> int:
    """How many runs to evolve at once (``__parallel_jobs_*__.num_workers``)."""
    block = cfg.get(spec.parallel_key, {})
    if not isinstance(block, dict):
        return 1
    return max(1, int(block.get("num_workers", 1)))


def resolve_device(requested: str | None, cfg: dict[str, Any]) -> str:
    import torch

    runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    device = requested or str(runtime.get("device", "auto"))
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_job_cfg(
    spec: DatasetSpec,
    cfg: dict[str, Any],
    *,
    results_dir: Path,
    device: str,
    test_run: bool = False,
) -> dict[str, Any]:
    evo = dict(cfg.get("evolution", {}) if isinstance(cfg.get("evolution"), dict) else {})
    runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    job_cfg = dict(evo)
    job_cfg["n_runs"] = 1
    job_cfg["results_dir"] = str(results_dir.resolve()) + os.sep
    job_cfg["device"] = device
    job_cfg["batch_size_eval"] = runtime.get(
        "batch_size_eval", 16 if spec.name == "imagenet" else 256
    )
    job_cfg["amp"] = bool(runtime.get("amp", spec.name == "imagenet"))
    job_cfg["turtle_delta_scale"] = float(runtime.get("turtle_delta_scale", 5.0))
    job_cfg["model_ckpt"] = resolve_model_ckpt(spec, cfg)
    if test_run:
        job_cfg["population_size"] = 10
        job_cfg["max_generations"] = 2
    return job_cfg


def _to_image_np(img: Any) -> np.ndarray:
    import torch

    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_samples(
    spec: DatasetSpec,
    turtle: ModuleType,
    cfg: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    imagenet_root: Path | None = None,
) -> list[tuple[int, int, np.ndarray]]:
    """Return (sample_idx, y_true, image_np) for each requested entry."""
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    out: list[tuple[int, int, np.ndarray]] = []

    if spec.name == "imagenet":
        root = imagenet_root or Path(
            str(paths.get("imagenet100_root") or paths.get("imagenet_root") or spec.data_dir)
        )
        if not root.is_absolute():
            root = (spec.dataset_dir / root).resolve()
        manifest_ready = [
            e
            for e in entries
            if "y_true" in e and _manifest_index(e) >= 0
        ]
        loader = getattr(turtle, "load_imagenet_val_manifest_samples", None)
        if loader is not None and manifest_ready and root.is_dir():
            rows = loader(root, manifest_ready, max_samples=len(manifest_ready))
            return [(int(idx), int(y), np.ascontiguousarray(arr, dtype=np.float32)) for idx, y, arr in rows]

        from imagenet100_dataset import ImageNet100AlibiDataset
        from imagenet_subset import alibi_rgb_transform

        if not root.is_dir():
            raise FileNotFoundError(
                f"ImageNet root not found: {root}. Pass --imagenet-root."
            )
        ds = ImageNet100AlibiDataset(root, "val", transform=alibi_rgb_transform())
        for e in entries:
            idx = _manifest_index(e)
            if idx < 0:
                idx = 0
            img_t, y_ds = ds[idx]
            y_true = int(e["y_true"]) if "y_true" in e else int(y_ds)
            out.append((idx, y_true, _to_image_np(img_t.squeeze())))
        return out

    data_dir = Path(str(paths.get("data_dir", spec.data_dir)))
    if not data_dir.is_absolute():
        data_dir = (spec.graco_dir / data_dir).resolve()
    download = bool(paths.get("download", True))
    raw_ds = turtle.load_dataset(data_dir=str(data_dir), download=download, train=False)
    for e in entries:
        idx = _manifest_index(e)
        if idx < 0:
            idx = 0
        img_t, y_ds = raw_ds[idx]
        y_true = int(e["y_true"]) if "y_true" in e else int(y_ds)
        img = _to_image_np(img_t)
        if spec.name == "mnist":
            img = np.ascontiguousarray(np.squeeze(img), dtype=np.float32)
        out.append((idx, y_true, img))
    return out


def evolve_one_run(
    spec: DatasetSpec,
    turtle: ModuleType,
    job_cfg: dict[str, Any],
    *,
    image: np.ndarray,
    label: int,
    sample_idx: int,
    run_id: int,
    seed: int,
) -> Path:
    results_dir = Path(job_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = results_dir / "evolution_trace.npz"

    orig_ge = turtle.algorithms.ge_eaSimpleWithElitism

    def _ge_wrapped(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        stats = kwargs.get("stats")
        if stats is not None:
            fields = list(getattr(stats, "fields", []) or [])
            if "min_obj" not in fields:
                stats.register("min_obj", _min_per_obj)
            if "avg_obj" not in fields:
                stats.register("avg_obj", _avg_per_obj)
        population, logbook = orig_ge(*args, **kwargs)
        write_evolution_trace(
            logbook,
            trace_path,
            seed=seed,
            sample_idx=sample_idx,
            y_true=label,
            run_id=run_id,
        )
        return population, logbook

    turtle.algorithms.ge_eaSimpleWithElitism = _ge_wrapped
    try:
        with _working_directory(spec.graco_dir), _locked_seed(seed):
            fd = {"decideTurtle": turtle.decideTurtle}
            turtle.evolveImage(job_cfg, function_dict=fd, image=image, label=int(label))
    finally:
        turtle.algorithms.ge_eaSimpleWithElitism = orig_ge

    if not trace_path.is_file():
        raise RuntimeError(f"evolution_trace.npz was not written: {trace_path}")
    return trace_path


def parallel_evolve_job(payload: dict[str, Any]) -> tuple[int, int, int, str, str]:
    """One run in a worker process. Returns (sample_idx, y_true, run_id, status, message)."""
    dataset = str(payload["dataset"])
    spec = SPECS[dataset]
    config_raw = payload.get("config")
    config_path = Path(config_raw) if config_raw else None
    cfg = load_experiment_config(spec, config_path)
    sample_idx = int(payload["sample_idx"])
    y_true = int(payload["y_true"])
    run_id = int(payload["run_id"])
    seed = int(payload["seed"])
    try:
        turtle = import_turtle(dataset)
        job_cfg = build_job_cfg(
            spec,
            cfg,
            results_dir=Path(payload["run_dir"]),
            device=str(payload["device"]),
            test_run=bool(payload.get("test_run", False)),
        )
        trace = evolve_one_run(
            spec,
            turtle,
            job_cfg,
            image=np.ascontiguousarray(payload["image"], dtype=np.float32),
            label=y_true,
            sample_idx=sample_idx,
            run_id=run_id,
            seed=seed,
        )
        return (sample_idx, y_true, run_id, "ok", str(trace))
    except Exception as exc:
        return (sample_idx, y_true, run_id, "error", str(exc))


def discover_traces(results_root: Path) -> list[Path]:
    if not results_root.is_dir():
        return []
    return sorted(results_root.rglob("evolution_trace.npz"))


def discover_final_generation_csvs(results_root: Path) -> list[Path]:
    """``{idx}_label_{y}/run_{r}/run_{r}_final_generation.csv`` (and similar)."""
    if not results_root.is_dir():
        return []
    return sorted(p for p in results_root.rglob("*_final_generation.csv") if p.is_file())


def parse_sample_folder(run_dir: Path) -> tuple[int, int, int]:
    """run_dir is .../{idx}_label_{y}/run_{r}."""
    run_name = run_dir.name
    sample_name = run_dir.parent.name
    run_id = int(run_name.split("_", 1)[1]) if "_" in run_name else 0
    if "_label_" in sample_name:
        left, right = sample_name.split("_label_", 1)
        return int(left), int(right), run_id
    return -1, -1, run_id
