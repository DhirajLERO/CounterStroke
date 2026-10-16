"""Draw a random grammar pool, score it in GPU batches, and keep three picks."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_GRACOIMAGE = Path(__file__).resolve().parents[1]
SEED = 10

# Forward-pass sizes. These are not evolutionary generations.
DEFAULT_SCORE_BATCH = {
    "mnist": 2048,
    "cifar10": 512,
    "imagenet100": 64,
}


def evaluation_budget(population_size: int, max_generations: int) -> int:
    """Programs per image. Elite size is ignored."""
    return int(population_size) * int(max_generations)


@dataclass
class Scored:
    o1: float
    o2: float
    o3: float
    pred_class: int
    pred_prob: float
    phenotype: str
    image: np.ndarray


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _drop_shared_imports() -> None:
    """Each turtle imports grape_v2 and its own explainers package."""
    for name in ("grape_v2", "algorithms_AG", "infer_utils", "config_loader"):
        sys.modules.pop(name, None)
    for name in list(sys.modules):
        if name == "explainers" or name.startswith("explainers."):
            sys.modules.pop(name, None)


def _prefer_dataset_root(root: Path) -> None:
    """Put this dataset's folder first so ``import explainers`` finds its package."""
    root_s = str(root.resolve())
    others = {
        str((_GRACOIMAGE / name).resolve())
        for name in ("MNIST", "Cifar10", "Imagenet")
    }
    sys.path[:] = [entry for entry in sys.path if entry not in others]
    sys.path.insert(0, root_s)


def load_backend(dataset: str):
    """Import one dataset turtle. Call this once per dataset, in order."""
    _drop_shared_imports()
    if dataset == "mnist":
        root = _GRACOIMAGE / "MNIST"
        graco = root / "graco"
        turtle_path = graco / "graco_v8_turtle.py"
        tag = "mnist_turtle_random"
    elif dataset == "cifar10":
        root = _GRACOIMAGE / "Cifar10"
        graco = root / "graco"
        turtle_path = graco / "graco_cifar_turtle.py"
        tag = "cifar_turtle_random"
    elif dataset == "imagenet100":
        root = _GRACOIMAGE / "Imagenet"
        graco = root / "graco"
        turtle_path = graco / "graco_inception_turtle.py"
        tag = "imagenet_turtle_random"
    else:
        raise ValueError(dataset)

    _prefer_dataset_root(root)
    sys.path.insert(0, str(graco))
    old = os.getcwd()
    os.chdir(graco)
    try:
        turtle = _load_module(tag, turtle_path)
        export = _load_module(f"export_{tag}", graco / "export_graco_strategies.py")
    finally:
        os.chdir(old)
    return turtle, export


def _individual_class(grape_mod, tag: str):
    from deap import base, creator

    fit_name = f"RandomFit_{tag}"
    ind_name = f"RandomInd_{tag}"
    if not hasattr(creator, fit_name):
        creator.create(fit_name, base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, ind_name):
        creator.create(ind_name, grape_mod.Individual, fitness=getattr(creator, fit_name))
    return getattr(creator, ind_name)


def _make_population(turtle, ind_class, evo: dict, n: int, function_dict: dict):
    if evo.get("random_initilisation"):
        raise RuntimeError("This ablation uses sensible initialisation. Leave random_initilisation false.")
    grape = turtle.grape
    return grape.sensible_initialisation(
        ind_class,
        n,
        bnf_grammar=turtle.BNF_GRAMMAR,
        min_init_depth=int(evo["min_init_tree_depth"]),
        max_init_depth=int(evo["max_init_tree_depth"]),
        codon_size=int(evo["codon_size"]),
        codon_consumption=evo["codon_consumption"],
        genome_representation=evo["genome_representation"],
        function_dict=function_dict,
    )


def _objectives(turtle, dataset: str, images: np.ndarray, factual, label: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "imagenet100":
        o1 = np.asarray(
            turtle.evaluate_probability_inception(images, label, _objectives.model, _objectives.batch, _objectives.device, amp=_objectives.amp),
            dtype=np.float64,
        )
        o2 = np.asarray(turtle.evaluate_sparsity_rgb(images, factual), dtype=np.float64)
        o3 = np.asarray(turtle.evaluate_proximity_rgb(images, factual), dtype=np.float64)
    else:
        o1 = np.asarray(
            turtle.evaluate_probability(images, label, _objectives.model, _objectives.batch, _objectives.device, amp=_objectives.amp),
            dtype=np.float64,
        )
        o2 = np.asarray(turtle.evaluate_sparsity(images, factual), dtype=np.float64)
        o3 = np.asarray(turtle.evaluate_proximity(images, factual), dtype=np.float64)
    penalized = o1 >= 0
    o2 = o2.copy()
    o3 = o3.copy()
    o2[penalized] = 1_000_000
    o3[penalized] = 1_000_000
    return o1, o2, o3


def _bind_model(model, batch: int, device: str, amp: bool) -> None:
    _objectives.model = model
    _objectives.batch = batch
    _objectives.device = device
    _objectives.amp = amp


def _render(turtle, dataset: str, phenotypes: list[str], factual, scale: float) -> np.ndarray:
    if dataset == "imagenet100":
        return turtle.process_phenotypes_for_images_rgb(
            phenotypes, factual, turtle_delta_scale=scale
        )
    return turtle.process_phenotypes_for_images(
        phenotypes, factual, turtle_delta_scale=scale
    )


def _report(turtle, images: np.ndarray):
    return turtle.model_report_batch(
        _objectives.model,
        images,
        device=_objectives.device,
        batch_size=_objectives.batch,
        amp=_objectives.amp,
    )


def _beats(new: Scored, old: Scored | None, key) -> bool:
    if old is None:
        return True
    return key(new) < key(old)


def _generation_row(gen: int, n_drawn: int, best: list[float], picks: dict[str, Scored | None]) -> dict[str, Any]:
    """One point on the same axes as Graco's per-generation objective log."""
    row: dict[str, Any] = {
        "gen": int(gen),
        "n_drawn": int(n_drawn),
        "min_o1": best[0],
        "min_o2": best[1],
        "min_o3": best[2],
    }
    for name, key in (
        ("pick_sparse", "graco_base_o2"),
        ("pick_prox", "graco_base_o3"),
        ("pick_50", "graco50"),
    ):
        scored = picks.get(key)
        for obj in ("o1", "o2", "o3"):
            row[f"{name}_{obj}"] = None if scored is None else float(getattr(scored, obj))
    return row


def _batch_state(picks, n_flips, n_mapped, drawn, history, best, next_gen) -> dict[str, Any]:
    return {
        "picks": picks,
        "n_flips": n_flips,
        "n_mapped": n_mapped,
        "n_drawn": drawn,
        "history": history,
        "best_obj": list(best),
        "next_gen": int(next_gen),
    }


def _absorb_score(best: list[float], picks: dict[str, Scored | None], scored: Scored, flipped: bool) -> None:
    best[0] = min(best[0], scored.o1)
    best[1] = min(best[1], scored.o2)
    best[2] = min(best[2], scored.o3)
    if not flipped:
        return
    if _beats(scored, picks.get("graco_base_o2"), lambda item: (item.o2, item.o3)):
        picks["graco_base_o2"] = scored
    if _beats(scored, picks.get("graco_base_o3"), lambda item: (item.o3, item.o2)):
        picks["graco_base_o3"] = scored
    if scored.pred_prob >= 0.5 and _beats(scored, picks.get("graco50"), lambda item: (item.o2, item.o3)):
        picks["graco50"] = scored


def _close_generations(
    cursor: int,
    population_size: int,
    n_programs: int,
    picks: dict[str, Scored | None],
    best: list[float],
    history: list[dict[str, Any]],
    next_gen: int,
) -> int:
    """Record every population boundary the draw cursor has reached."""
    if population_size <= 0:
        return next_gen
    while cursor >= (next_gen + 1) * population_size and (next_gen + 1) * population_size <= n_programs:
        boundary = (next_gen + 1) * population_size
        history.append(_generation_row(next_gen, boundary, best, picks))
        next_gen += 1
    return next_gen


def sample_one_image(
    turtle,
    export,
    dataset: str,
    factual: np.ndarray,
    label: int,
    evo: dict,
    n_programs: int,
    score_batch: int,
    scale: float,
    population_size: int,
    resume: dict[str, Any] | None = None,
    on_batch=None,
) -> dict[str, Any]:
    """Draw n_programs programs and apply the three rules to the whole set."""
    grape = turtle.grape
    ind_class = _individual_class(grape, dataset)
    function_dict = {"decideTurtle": turtle.decideTurtle}
    population_size = int(population_size)
    if resume is None:
        picks: dict[str, Scored | None] = {"graco_base_o2": None, "graco_base_o3": None, "graco50": None}
        n_flips = 0
        n_mapped = 0
        drawn = 0
        history: list[dict[str, Any]] = []
        best = [math.inf, math.inf, math.inf]
        next_gen = 0
    else:
        picks = resume["picks"]
        n_flips = int(resume["n_flips"])
        n_mapped = int(resume["n_mapped"])
        drawn = int(resume["n_drawn"])
        history = list(resume.get("history") or [])
        stored_best = resume.get("best_obj")
        best = [math.inf, math.inf, math.inf] if not stored_best else [float(v) for v in stored_best]
        if "next_gen" in resume:
            next_gen = int(resume["next_gen"])
        else:
            next_gen = drawn // population_size if population_size else 0
    while drawn < n_programs:
        n = min(int(score_batch), n_programs - drawn)
        population = _make_population(turtle, ind_class, evo, n, function_dict)
        if not population:
            raise RuntimeError("Grammar initialisation returned no programs")
        drawn_before = drawn
        drawn += len(population)
        usable = [ind for ind in population if not getattr(ind, "invalid", False) and ind.phenotype]
        if not usable:
            next_gen = _close_generations(
                drawn, population_size, n_programs, picks, best, history, next_gen
            )
            if on_batch is not None:
                on_batch(_batch_state(picks, n_flips, n_mapped, drawn, history, best, next_gen))
            continue
        phenotypes = [ind.phenotype for ind in usable]
        try:
            images = _render(turtle, dataset, phenotypes, factual, scale)
        except Exception as exc:
            print(f"  render batch failed ({exc}); scoring programs one by one", flush=True)
            images_list = []
            kept_phenos = []
            for phenotype in phenotypes:
                try:
                    one = _render(turtle, dataset, [phenotype], factual, scale)
                except Exception:
                    continue
                images_list.append(one[0])
                kept_phenos.append(phenotype)
            if not images_list:
                next_gen = _close_generations(
                    drawn, population_size, n_programs, picks, best, history, next_gen
                )
                if on_batch is not None:
                    on_batch(_batch_state(picks, n_flips, n_mapped, drawn, history, best, next_gen))
                continue
            images = np.stack(images_list, axis=0)
            phenotypes = kept_phenos
        n_mapped += len(phenotypes)
        pred_cls, pred_prob, _probs = _report(turtle, images)
        o1, o2, o3 = _objectives(turtle, dataset, images, factual, int(label))
        scored_rows: list[Scored] = []
        flipped_rows: list[bool] = []
        batch_rows = []
        by_key: dict[tuple, Scored] = {}
        for i, phenotype in enumerate(phenotypes):
            scored = Scored(
                o1=float(o1[i]),
                o2=float(o2[i]),
                o3=float(o3[i]),
                pred_class=int(pred_cls[i]),
                pred_prob=float(pred_prob[i]),
                phenotype=phenotype,
                image=np.array(images[i], copy=True),
            )
            scored_rows.append(scored)
            flipped = int(pred_cls[i]) != int(label)
            flipped_rows.append(flipped)
            if not flipped:
                continue
            n_flips += 1
            batch_rows.append(
                export.Row(
                    o1=scored.o1,
                    o2=scored.o2,
                    o3=scored.o3,
                    original_class=int(label),
                    pred_class=scored.pred_class,
                    pred_prob=scored.pred_prob,
                    phenotype=scored.phenotype,
                )
            )
            by_key[(scored.phenotype, scored.o2, scored.o3, scored.pred_prob)] = scored
        score_cursor = 0
        cursor = drawn_before
        for ind in population:
            cursor += 1
            phenotype = getattr(ind, "phenotype", None)
            mapped = (not getattr(ind, "invalid", False)) and bool(phenotype)
            if (
                mapped
                and score_cursor < len(phenotypes)
                and phenotype == phenotypes[score_cursor]
            ):
                _absorb_score(best, picks, scored_rows[score_cursor], flipped_rows[score_cursor])
                score_cursor += 1
            next_gen = _close_generations(
                cursor, population_size, n_programs, picks, best, history, next_gen
            )
        picks = _apply_rules(export, picks, batch_rows, by_key)
        print(
            f"  drawn {drawn}/{n_programs} mapped {n_mapped} flips {n_flips}",
            flush=True,
        )
        if on_batch is not None:
            on_batch(_batch_state(picks, n_flips, n_mapped, drawn, history, best, next_gen))
    return {
        "picks": picks,
        "n_flips": n_flips,
        "n_mapped": n_mapped,
        "n_drawn": drawn,
        "history": history,
    }


def _apply_rules(export, picks: dict[str, Scored | None], batch_rows: list, by_key: dict) -> dict[str, Scored | None]:
    """Apply the exporter pick rules to the current winners plus this batch."""
    pool = []
    for scored in picks.values():
        if scored is None:
            continue
        by_key[(scored.phenotype, scored.o2, scored.o3, scored.pred_prob)] = scored
        pool.append(
            export.Row(
                o1=scored.o1,
                o2=scored.o2,
                o3=scored.o3,
                original_class=-1,
                pred_class=scored.pred_class,
                pred_prob=scored.pred_prob,
                phenotype=scored.phenotype,
            )
        )
    pool.extend(batch_rows)
    chosen = {
        "graco_base_o2": export._pick_min_o2_then_o3(pool),
        "graco_base_o3": export._pick_min_o3_then_o2(pool),
        "graco50": export._pick_graco50(pool),
    }
    out: dict[str, Scored | None] = {}
    for key, row in chosen.items():
        if row is None:
            out[key] = None
            continue
        lookup = (row.phenotype, row.o2, row.o3, row.pred_prob)
        scored = by_key.get(lookup)
        if scored is None:
            raise KeyError(f"Picked program {key} has no rendered image")
        out[key] = scored
    return out
