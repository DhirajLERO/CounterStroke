"""Graco evolution from a blank canvas with O1 = p(label) - p(wanted_class)."""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .graco_reports import write_final_generation_csv
from .graco_target_fitness import evaluate_probability_target, resolve_inference_device


@dataclass
class TurtleHooks:
    """Callables imported from a dataset graco turtle module."""

    decide_turtle: Callable
    process_phenotypes: Callable
    model_report_batch: Callable
    evaluate_sparsity: Callable
    evaluate_proximity: Callable
    to_display_01: Callable
    bnf_grammar: Any
    fitness_creator_name: str
    individual_creator_name: str


def load_turtle_hooks(graco_dir: Path, *, dataset: str) -> TurtleHooks:
    """Import turtle helpers without modifying dataset source files."""
    graco_dir = graco_dir.resolve()
    pkg_dir = graco_dir.parent
    for p in (str(pkg_dir), str(graco_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    old_cwd = os.getcwd()
    os.chdir(graco_dir)
    try:
        if dataset == "mnist":
            mod_name = "graco_v8_turtle"
            spec = importlib.util.spec_from_file_location(
                mod_name, graco_dir / "graco_v8_turtle.py"
            )
        elif dataset == "cifar10":
            mod_name = "graco_cifar_turtle"
            spec = importlib.util.spec_from_file_location(
                mod_name, graco_dir / "graco_cifar_turtle.py"
            )
        elif dataset == "imagenet100":
            mod_name = "graco_inception_turtle"
            spec = importlib.util.spec_from_file_location(
                mod_name, graco_dir / "graco_inception_turtle.py"
            )
        else:
            raise ValueError(f"Unknown dataset {dataset!r}")

        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        import grape_v2 as grape  # noqa: WPS433

        if dataset == "imagenet100":
            grammar = grape.Grammar(str(graco_dir / "grammar" / "AG_turtle_rgb.bnf"))
            process_fn = mod.process_phenotypes_for_images_rgb
            to_01 = mod.alibi_chw_to_01
            fit_name, ind_name = "FitnessMinBlankIV3", "IndividualBlankIV3"
        elif dataset == "cifar10":
            grammar = grape.Grammar(str(graco_dir / "grammar" / "AG_turtle_cifar.bnf"))
            process_fn = mod.process_phenotypes_for_images
            from cifar_transforms import alibi_to_01_rgb  # noqa: WPS433

            to_01 = alibi_to_01_rgb
            fit_name, ind_name = "FitnessMinBlank", "IndividualBlank"
        else:
            grammar = grape.Grammar(str(graco_dir / "grammar" / "AG_turtle.bnf"))
            process_fn = mod.process_phenotypes_for_images

            def _mnist_01(img_hw: np.ndarray) -> np.ndarray:
                return np.clip(img_hw.astype(np.float32) + 0.5, 0.0, 1.0)

            to_01 = _mnist_01
            fit_name, ind_name = "FitnessMinBlank", "IndividualBlank"

        return TurtleHooks(
            decide_turtle=mod.decideTurtle,
            process_phenotypes=process_fn,
            model_report_batch=mod.model_report_batch,
            evaluate_sparsity=(
                mod.evaluate_sparsity_rgb if dataset == "imagenet100" else mod.evaluate_sparsity
            ),
            evaluate_proximity=(
                mod.evaluate_proximity_rgb if dataset == "imagenet100" else mod.evaluate_proximity
            ),
            to_display_01=to_01,
            bnf_grammar=grammar,
            fitness_creator_name=fit_name,
            individual_creator_name=ind_name,
        )
    finally:
        os.chdir(old_cwd)


def fitness_eval_target(
    individuals,
    points,
    image: np.ndarray,
    label: int,
    wanted_class: int,
    model,
    batch_size: int,
    device,
    amp: bool,
    turtle_delta_scale: float,
    hooks: TurtleHooks,
) -> tuple[list[float], list[float], list[float]]:
    phenos = [
        ind if isinstance(ind, str) else ind.phenotype
        for ind in individuals
        if not ind.fitness.valid
    ]
    if phenos:
        individuals_np = hooks.process_phenotypes(
            phenos, image, turtle_delta_scale=turtle_delta_scale
        )
        o1_list = evaluate_probability_target(
            individuals_np,
            label,
            wanted_class,
            hooks.model_report_batch,
            model,
            batch_size,
            device,
            amp=amp,
        )
        o2_list = hooks.evaluate_sparsity(individuals_np, image)
        o3_list = hooks.evaluate_proximity(individuals_np, image)
    else:
        o1_list, o2_list, o3_list = [], [], []

    o1_out, o2_out, o3_out = [], [], []
    j = 0
    for ind in individuals:
        if ind.fitness.valid:
            o1_out.append(1000.0)
            o2_out.append(1000.0)
            o3_out.append(1000.0)
        else:
            o1_score = float(o1_list[j])
            o1_out.append(o1_score)
            if o1_score >= 0:
                o2_out.append(1_000_000.0)
                o3_out.append(1_000_000.0)
            else:
                o2_out.append(float(o2_list[j]))
                o3_out.append(float(o3_list[j]))
            j += 1
    return o1_out, o2_out, o3_out


def evolve_blank_canvas(
    *,
    hooks: TurtleHooks,
    model,
    blank_image: np.ndarray,
    label: int,
    wanted_class: int,
    cfg: dict,
    results_dir: Path,
    device: str | torch.device | None = None,
) -> Path:
    """Run one NSGA-II evolution; write run_0_final_generation.csv."""
    import algorithms_AG as algorithms  # noqa: WPS433
    import grape_v2 as grape  # noqa: WPS433
    from deap import base, creator, tools  # noqa: WPS433

    graco_dir = Path(cfg["_graco_dir"]).resolve()
    old_cwd = os.getcwd()
    os.chdir(graco_dir)
    try:
        defaults = {
            "population_size": 1000,
            "max_generations": 500,
            "p_crossover": 0.9,
            "p_mutation": 0.01,
            "elite_size": 3,
            "halloffame_size": 3,
            "tournsize": 10,
            "min_init_genome_length": 30,
            "max_init_genome_length": 50,
            "random_initilisation": False,
            "max_init_tree_depth": 20,
            "min_init_tree_depth": 7,
            "max_tree_depth": 60,
            "max_wraps": 5,
            "codon_size": 1024,
            "codon_consumption": "lazy",
            "genome_representation": "list",
            "max_genome_length": None,
            "report_items": [
                "gen",
                "invalid",
                "avg",
                "std",
                "min",
                "max",
                "fitness_test",
                "best_ind_length",
                "avg_length",
                "best_ind_nodes",
                "avg_nodes",
                "best_ind_depth",
                "avg_depth",
                "avg_used_codons",
                "best_ind_used_codons",
                "structural_diversity",
                "selection_time",
                "generation_time",
            ],
            "n_runs": 1,
        }
        merged = {**defaults, **cfg}
        dev = resolve_inference_device(
            model, device if device is not None else merged.get("device")
        )
        batch_size_eval = int(merged.get("batch_size_eval", 32))
        amp = bool(merged.get("amp", False))
        turtle_delta_scale = float(merged.get("turtle_delta_scale", 5.0))

        fit_name = hooks.fitness_creator_name
        ind_name = hooks.individual_creator_name
        if not hasattr(creator, fit_name):
            creator.create(fit_name, base.Fitness, weights=(-1.0, -1.0, -1.0))
        if not hasattr(creator, ind_name):
            creator.create(ind_name, grape.Individual, fitness=getattr(creator, fit_name))

        toolbox = base.Toolbox()
        toolbox.register(
            "populationCreator",
            grape.sensible_initialisation,
            getattr(creator, ind_name),
        )
        toolbox.register(
            "evaluate",
            fitness_eval_target,
            image=blank_image,
            label=int(label),
            wanted_class=int(wanted_class),
            model=model,
            batch_size=batch_size_eval,
            device=dev,
            amp=amp,
            turtle_delta_scale=turtle_delta_scale,
            hooks=hooks,
        )
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("mate", grape.crossover_onepoint)
        toolbox.register("mutate", grape.mutation_int_flip_per_codon)

        function_dict = {"decideTurtle": hooks.decide_turtle}
        results_dir.mkdir(parents=True, exist_ok=True)

        for run_i in range(int(merged["n_runs"])):
            np.random.seed(10 + run_i)
            random.seed(10 + run_i)

            if merged["random_initilisation"]:
                population = toolbox.populationCreator(
                    pop_size=merged["population_size"],
                    bnf_grammar=hooks.bnf_grammar,
                    min_init_genome_length=merged["min_init_genome_length"],
                    max_init_genome_length=merged["max_init_genome_length"],
                    max_init_depth=merged["max_tree_depth"],
                    codon_size=merged["codon_size"],
                    codon_consumption=merged["codon_consumption"],
                    genome_representation=merged["genome_representation"],
                )
            else:
                population = toolbox.populationCreator(
                    pop_size=merged["population_size"],
                    bnf_grammar=hooks.bnf_grammar,
                    min_init_depth=merged["min_init_tree_depth"],
                    max_init_depth=merged["max_init_tree_depth"],
                    codon_size=merged["codon_size"],
                    codon_consumption=merged["codon_consumption"],
                    genome_representation=merged["genome_representation"],
                    function_dict=function_dict,
                )

            hof = tools.HallOfFame(merged["halloffame_size"])
            stats = tools.Statistics(key=lambda ind: ind.fitness.values)
            stats.register("avg", np.nanmean)
            stats.register("std", np.nanstd)
            stats.register("min", np.nanmin)
            stats.register("max", np.nanmax)

            population, _logbook = algorithms.ge_eaSimpleWithElitism(
                population,
                toolbox,
                cxpb=merged["p_crossover"],
                mutpb=merged["p_mutation"],
                ngen=merged["max_generations"],
                elite_size=merged["elite_size"],
                bnf_grammar=hooks.bnf_grammar,
                codon_size=merged["codon_size"],
                max_tree_depth=merged["max_tree_depth"],
                max_genome_length=merged["max_genome_length"],
                points_train=None,
                points_test=None,
                codon_consumption=merged["codon_consumption"],
                report_items=merged["report_items"],
                genome_representation=merged["genome_representation"],
                invalidate_max_depth=False,
                stats=stats,
                halloffame=hof,
                verbose=False,
                function_dict=function_dict,
            )

            phenotypes_all = [ind.phenotype for ind in population]
            imgs_batch = hooks.process_phenotypes(
                phenotypes_all, blank_image, turtle_delta_scale=turtle_delta_scale
            )
            pred_cls, pred_prob, probs = hooks.model_report_batch(
                model, imgs_batch, device=dev, batch_size=batch_size_eval, amp=amp
            )
            p_original = probs[:, int(label)]
            o1_arr = np.array([ind.fitness.values[0] for ind in population], dtype=np.float32)
            o2_arr = np.array([ind.fitness.values[1] for ind in population], dtype=np.float32)
            o3_arr = np.array([ind.fitness.values[2] for ind in population], dtype=np.float32)

            csv_path = results_dir / f"run_{run_i}_final_generation.csv"
            write_final_generation_csv(
                out_csv_path=str(csv_path),
                phenotypes=phenotypes_all,
                o1=o1_arr,
                o2=o2_arr,
                o3=o3_arr,
                original_class=int(label),
                p_original=p_original,
                pred_class=pred_cls,
                pred_prob=pred_prob,
            )
        return results_dir / "run_0_final_generation.csv"
    finally:
        os.chdir(old_cwd)
