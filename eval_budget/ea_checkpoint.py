"""In-memory checkpointing wrapper around ge_eaSimpleWithElitism."""

from __future__ import annotations

import contextvars
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from snapshots import (
    capture_rng,
    clear_partial,
    load_checkpoint,
    population_csv_path,
    restore_rng,
    save_checkpoint,
    write_objectives,
    write_population_csv,
)

CURRENT: contextvars.ContextVar["BudgetContext | None"] = contextvars.ContextVar(
    "eval_budget_context",
    default=None,
)


@dataclass
class BudgetContext:
    dataset: str
    run_dir: Path
    seed: int
    population_size: int
    max_generations: int
    snapshot_gens: tuple[int, ...]
    label: int
    image: np.ndarray
    turtle: Any
    device: str
    batch_size: int
    amp: bool
    turtle_delta_scale: float
    model_ckpt: str
    model: Any = field(default=None, repr=False)


def install(turtle: ModuleType) -> None:
    """Replace the turtle's EA lookup. Call once per process, after import."""
    algorithms = turtle.algorithms
    current = algorithms.ge_eaSimpleWithElitism
    if getattr(current, "_eval_budget_wrapped", False):
        return

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        ctx = CURRENT.get()
        if ctx is None:
            return current(*args, **kwargs)
        return _run(algorithms, ctx, *args, **kwargs)

    wrapped._eval_budget_wrapped = True  # type: ignore[attr-defined]
    algorithms.ge_eaSimpleWithElitism = wrapped


def _load_model(ctx: BudgetContext) -> Any:
    if ctx.model is not None:
        return ctx.model
    import torch

    if ctx.dataset == "mnist":
        model = ctx.turtle.MNISTNet().to(ctx.device)
        model.load_state_dict(torch.load(ctx.model_ckpt, map_location=ctx.device))
    elif ctx.dataset == "cifar":
        model = ctx.turtle.CifarResNet().to(ctx.device)
        model.load_state_dict(torch.load(ctx.model_ckpt, map_location=ctx.device))
    else:
        model = ctx.turtle.load_task_classifier(ctx.model_ckpt, ctx.device)
    model.eval()
    ctx.model = model
    return model


def _render(ctx: BudgetContext, phenotypes: list[str]) -> np.ndarray:
    scale = float(ctx.turtle_delta_scale)
    if ctx.dataset == "imagenet":
        return ctx.turtle.process_phenotypes_for_images_rgb(
            phenotypes, ctx.image, turtle_delta_scale=scale
        )
    return ctx.turtle.process_phenotypes_for_images(
        phenotypes, ctx.image, turtle_delta_scale=scale
    )


def _snapshot(ctx: BudgetContext, generation: int, population: list[Any]) -> None:
    if int(generation) not in ctx.snapshot_gens:
        return
    path = population_csv_path(ctx.run_dir, generation)
    if path.is_file():
        return
    phenotypes = [str(getattr(ind, "phenotype", "") or "") for ind in population]
    o1 = np.array([ind.fitness.values[0] for ind in population], dtype=np.float64)
    o2 = np.array([ind.fitness.values[1] for ind in population], dtype=np.float64)
    o3 = np.array([ind.fitness.values[2] for ind in population], dtype=np.float64)
    model = _load_model(ctx)
    images = _render(ctx, phenotypes)
    pred_class, pred_prob, probs = ctx.turtle.model_report_batch(
        model,
        images,
        device=ctx.device,
        batch_size=int(ctx.batch_size),
        amp=bool(ctx.amp),
    )
    p_original = np.asarray(probs[:, int(ctx.label)], dtype=np.float64)
    write_population_csv(
        path,
        phenotypes=phenotypes,
        o1=o1,
        o2=o2,
        o3=o3,
        original_class=int(ctx.label),
        p_original=p_original,
        pred_class=np.asarray(pred_class),
        pred_prob=np.asarray(pred_prob, dtype=np.float64),
    )
    print(f"  snapshot gen {generation} -> {path}", flush=True)


def _commit(
    ctx: BudgetContext,
    *,
    generation: int,
    population: list[Any],
    old_population: list[Any] | None,
    halloffame: Any,
    logbook: Any,
) -> None:
    write_objectives(ctx.run_dir, logbook)
    _snapshot(ctx, generation, population)
    save_checkpoint(
        ctx.run_dir,
        {
            "completed_gen": int(generation),
            "population": population,
            "old_population": old_population,
            "hof_items": list(halloffame.items),
            "hof_keys": list(halloffame.keys),
            "logbook": logbook,
            "rng": capture_rng(),
            "seed": int(ctx.seed),
            "population_size": int(ctx.population_size),
            "max_generations": int(ctx.max_generations),
        },
    )


def _evaluate(toolbox: Any, pending: list[Any], points_train: Any) -> tuple[Any, Any, Any]:
    if not pending:
        return [], [], []
    return toolbox.evaluate(pending, points_train)


def _restore_hof(halloffame: Any, payload: dict[str, Any]) -> None:
    halloffame.items[:] = list(payload["hof_items"])
    halloffame.keys[:] = list(payload["hof_keys"])


def _diversity(valid: list[Any], population: list[Any], report_items: list[str], points_train: Any) -> tuple[float, float, float]:
    list_structures = []
    list_fitnesses = []
    behaviours = None
    if "behavioural_diversity" in report_items and valid:
        behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
    for idx, ind in enumerate(valid):
        list_structures.append(str(ind.structure))
        if "fitness_diversity" in report_items:
            list_fitnesses.append(str(ind.fitness.values[0]))
        if behaviours is not None:
            behaviours[idx, :] = ind.fitness_each_sample
    structural = len(np.unique(list_structures, return_counts=False)) / len(population)
    if "fitness_diversity" in report_items:
        fitness = len(np.unique(list_fitnesses, return_counts=False)) / (len(points_train[1]) + 1)
    else:
        fitness = 0
    if behaviours is not None:
        behavioural = len(np.unique(behaviours, axis=0)) / len(population)
    else:
        behavioural = 0
    return structural, fitness, behavioural


def _lengths(valid: list[Any]) -> tuple[float, float, float, float]:
    avg_length = sum(len(ind.genome) for ind in valid) / len(valid)
    avg_nodes = sum(ind.nodes for ind in valid) / len(valid)
    avg_depth = sum(ind.depth for ind in valid) / len(valid)
    avg_used = sum(ind.used_codons for ind in valid) / len(valid)
    return avg_length, avg_nodes, avg_depth, avg_used


def _valid_inds(population: list[Any]) -> tuple[list[Any], list[Any], int]:
    valid0 = [ind for ind in population if not ind.invalid]
    valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
    if len(valid0) != len(valid):
        warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid them.")
    invalid = len(population) - len(valid0)
    return valid0, valid, invalid


def _record(
    logbook: Any,
    *,
    gen: int,
    invalid: int,
    record: dict[str, Any],
    points_test: Any,
    fitness_test: Any,
    best_ind_length: int,
    avg_length: float,
    best_ind_nodes: Any,
    avg_nodes: float,
    best_ind_depth: Any,
    avg_depth: float,
    avg_used_codons: float,
    best_ind_used_codons: Any,
    behavioural_diversity: float,
    structural_diversity: float,
    fitness_diversity: float,
    selection_time: float,
    generation_time: float,
) -> None:
    common = dict(
        best_ind_length=best_ind_length,
        avg_length=avg_length,
        best_ind_nodes=best_ind_nodes,
        avg_nodes=avg_nodes,
        best_ind_depth=best_ind_depth,
        avg_depth=avg_depth,
        avg_used_codons=avg_used_codons,
        best_ind_used_codons=best_ind_used_codons,
        behavioural_diversity=behavioural_diversity,
        structural_diversity=structural_diversity,
        fitness_diversity=fitness_diversity,
        selection_time=selection_time,
        generation_time=generation_time,
    )
    if points_test:
        logbook.record(gen=gen, invalid=invalid, **record, fitness_test=fitness_test, **common)
    else:
        logbook.record(gen=gen, invalid=invalid, **record, **common)


def _run(
    algorithms: ModuleType,
    ctx: BudgetContext,
    population: list[Any],
    toolbox: Any,
    cxpb: float,
    mutpb: float,
    ngen: int,
    elite_size: int,
    bnf_grammar: Any,
    codon_size: int,
    max_tree_depth: int,
    max_genome_length: Any = None,
    points_train: Any = None,
    points_test: Any = None,
    codon_consumption: str = "eager",
    report_items: list[str] | None = None,
    genome_representation: str = "list",
    invalidate_max_depth: bool = False,
    stats: Any = None,
    halloffame: Any = None,
    verbose: bool = False,
    function_dict: Any = None,
) -> tuple[list[Any], Any]:
    if halloffame is None:
        raise RuntimeError("Budget evolution requires a hall of fame.")
    if int(ngen) != int(ctx.max_generations):
        raise RuntimeError(
            f"EA ngen={ngen} does not match the budget config max_generations={ctx.max_generations}."
        )
    report_items = list(report_items or [])
    from deap import tools

    saved = _load_matching_checkpoint(ctx)

    if saved is None:
        logbook = tools.Logbook()
        logbook.header = [
            "gen",
            "invalid",
            *(stats.fields if stats else []),
            "best_ind_length",
            "avg_length",
            "best_ind_nodes",
            "avg_nodes",
            "best_ind_depth",
            "avg_depth",
            "avg_used_codons",
            "best_ind_used_codons",
            "behavioural_diversity",
            "structural_diversity",
            "fitness_diversity",
            "selection_time",
            "generation_time",
        ]
        start_gen = time.time()
        pending = [ind for ind in population if not ind.fitness.valid]
        o1, o2, o3 = _evaluate(toolbox, pending, points_train)
        cursor = 0
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = (o1[cursor], o2[cursor], o3[cursor])
                cursor += 1
        _valid0, valid, invalid = _valid_inds(population)
        structural, fitness_div, behavioural = _diversity(valid, population, report_items, points_train)
        halloffame.update(valid)
        best = halloffame.items[0]
        if not verbose:
            print("gen =", 0, ", Best fitness =", best.fitness.values, flush=True)
        avg_length, avg_nodes, avg_depth, avg_used = _lengths(valid)
        record = stats.compile(population) if stats else {}
        _record(
            logbook,
            gen=0,
            invalid=invalid,
            record=record,
            points_test=points_test,
            fitness_test=np.nan,
            best_ind_length=len(best.genome),
            avg_length=avg_length,
            best_ind_nodes=best.nodes,
            avg_nodes=avg_nodes,
            best_ind_depth=best.depth,
            avg_depth=avg_depth,
            avg_used_codons=avg_used,
            best_ind_used_codons=best.used_codons,
            behavioural_diversity=behavioural,
            structural_diversity=structural,
            fitness_diversity=fitness_div,
            selection_time=0,
            generation_time=time.time() - start_gen,
        )
        old_population = None
        _commit(
            ctx,
            generation=0,
            population=population,
            old_population=old_population,
            halloffame=halloffame,
            logbook=logbook,
        )
        next_gen = 1
    else:
        population[:] = list(saved["population"])
        old_population = saved.get("old_population")
        _restore_hof(halloffame, saved)
        logbook = saved["logbook"]
        restore_rng(saved["rng"])
        write_objectives(ctx.run_dir, logbook)
        next_gen = int(saved["completed_gen"]) + 1
        _snapshot(ctx, int(saved["completed_gen"]), population)
        print(
            f"  resume {ctx.run_dir.name} at generation {next_gen} "
            f"(completed {saved['completed_gen']})",
            flush=True,
        )
        if next_gen > int(ngen):
            return population, logbook

    for gen in range(next_gen, int(ngen) + 1):
        start_gen = time.time()
        _valid0, valid, _invalid_before = _valid_inds(population)
        start = time.time()
        if gen == 1:
            offspring = toolbox.select(valid, len(population) - elite_size)
        else:
            offspring = toolbox.select(valid + old_population, len(population) - elite_size)
        selection_time = time.time() - start
        offspring = algorithms.varAnd(
            offspring,
            toolbox,
            cxpb,
            mutpb,
            bnf_grammar,
            codon_size,
            max_tree_depth,
            codon_consumption,
            invalidate_max_depth,
            genome_representation,
            max_genome_length,
            function_dict,
        )
        pending = [ind for ind in offspring if not ind.fitness.valid]
        o1, o2, o3 = _evaluate(toolbox, pending, points_train)
        cursor = 0
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = (o1[cursor], o2[cursor], o3[cursor])
                cursor += 1

        old_population = [ind for ind in population if not ind.invalid]
        old_population = [ind for ind in old_population if not math.isnan(ind.fitness.values[0])]
        population[:] = offspring
        for i in range(elite_size):
            population.append(halloffame.items[i])

        _valid0, valid, invalid = _valid_inds(population)
        structural, fitness_div, behavioural = _diversity(valid, population, report_items, points_train)
        halloffame.update(valid)
        best = halloffame.items[0]
        fitness_test = np.nan
        if points_test and gen == ngen:
            fitness_test = toolbox.evaluate(halloffame.items[0], points_test)[0]
        if not verbose:
            print(
                "gen =",
                gen,
                ", Best fitness =",
                best.fitness.values,
                ", Number of invalids =",
                invalid,
                flush=True,
            )
        avg_length, avg_nodes, avg_depth, avg_used = _lengths(valid)
        record = stats.compile(population) if stats else {}
        _record(
            logbook,
            gen=gen,
            invalid=invalid,
            record=record,
            points_test=points_test,
            fitness_test=fitness_test,
            best_ind_length=len(best.genome),
            avg_length=avg_length,
            best_ind_nodes=best.nodes,
            avg_nodes=avg_nodes,
            best_ind_depth=best.depth,
            avg_depth=avg_depth,
            avg_used_codons=avg_used,
            best_ind_used_codons=best.used_codons,
            behavioural_diversity=behavioural,
            structural_diversity=structural,
            fitness_diversity=fitness_div,
            selection_time=selection_time,
            generation_time=time.time() - start_gen,
        )
        _commit(
            ctx,
            generation=gen,
            population=population,
            old_population=old_population,
            halloffame=halloffame,
            logbook=logbook,
        )

    return population, logbook


def _load_matching_checkpoint(ctx: BudgetContext) -> dict[str, Any] | None:
    path = ctx.run_dir / "checkpoint.pkl"
    if not path.is_file():
        return None
    try:
        payload = load_checkpoint(ctx.run_dir)
    except Exception as exc:
        stale = ctx.run_dir / "checkpoint_unreadable.pkl"
        path.replace(stale)
        clear_partial(ctx.run_dir)
        print(f"  ignoring unreadable checkpoint ({exc}); moved to {stale.name}", flush=True)
        return None
    if payload is None:
        return None
    same = (
        int(payload.get("seed", -1)) == int(ctx.seed)
        and int(payload.get("population_size", -1)) == int(ctx.population_size)
        and int(payload.get("max_generations", -1)) == int(ctx.max_generations)
    )
    if not same:
        stale = ctx.run_dir / "checkpoint_mismatch.pkl"
        path.replace(stale)
        clear_partial(ctx.run_dir)
        print(f"  checkpoint does not match this job; moved to {stale.name}", flush=True)
        return None
    return payload
