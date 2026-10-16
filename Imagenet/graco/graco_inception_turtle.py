"""GracoImage on ImageNet val with RGB turtle grammar and Inception v3."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import copy
import csv
import json
import math
import os
import pickle
import random
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

_INCEPTION_DIR = Path(__file__).resolve().parent
_GRACO_ROOT = _INCEPTION_DIR.parent
_REPO_ROOT = _GRACO_ROOT.parent

for p in (_REPO_ROOT, _GRACO_ROOT, _INCEPTION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_mpl_dir = _INCEPTION_DIR / ".mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")

import algorithms_AG as algorithms
import grape_v2 as grape
import numpy as np
from common.objective_generation_log import (
    GENERATION_LOG_COLUMNS,
    evolution_seed,
    objective_row_suffix,
    register_objective_stats,
    scheduled_run_ids,
    write_objectives_by_generation,
)
import torch
from deap import base, creator, tools
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

try:
    from config_loader import load_config
except ImportError:
    load_config = None  # type: ignore[misc, assignment]

from apply_turtle_rgb import (
    alibi_chw_to_hwc_01,
    process_phenotypes_for_images_rgb,
)
from imagenet_subset import default_imagenet_root, load_imagenet_val_manifest_samples
from inception_infer import load_task_classifier, predict_probs_numpy_inception
from io_reports import write_final_generation_csv, write_run_summary_npz

x_max = 299
y_max = 299

GRAMMAR_FILE = "AG_turtle_rgb.bnf"
BNF_GRAMMAR = grape.Grammar(str(_INCEPTION_DIR / "grammar" / GRAMMAR_FILE))


def alibi_chw_to_01(img_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) Alibi -> (3,H,W) in [0,1]."""
    return np.clip(img_chw + 0.5, 0.0, 1.0).astype(np.float32, copy=False)


def _resolve_model_ckpt(cfg: dict) -> str:
    p = cfg.get("model_ckpt") or str(_GRACO_ROOT / "models" / "imagenet100_inception.pt")
    return str(Path(p).expanduser().resolve())


def model_report_batch(model, imgs_np, device="cpu", batch_size=32, amp=False):
    return predict_probs_numpy_inception(
        model, imgs_np, device=device, batch_size=batch_size, amp=amp
    )


def save_detailed_report_pdf(
    population,
    original_img_np,
    out_pdf_path,
    label_true,
    top_n=5,
    n_fronts=3,
    ncols=8,
    device="cpu",
    batch_size=32,
    amp=False,
    turtle_delta_scale: float = 5.0,
    model_ckpt: str | None = None,
):
    ckpt = model_ckpt or str(_GRACO_ROOT / "models" / "imagenet100_inception.pt")
    model = load_task_classifier(ckpt, device)
    pop_sorted_o1 = sorted(population, key=lambda ind: ind.fitness.values[0])
    pop_sorted_o2 = sorted(population, key=lambda ind: ind.fitness.values[1])
    pop_sorted_o3 = sorted(population, key=lambda ind: ind.fitness.values[2])
    top_o1 = pop_sorted_o1[:top_n]
    top_o2 = pop_sorted_o2[:top_n]
    top_o3 = pop_sorted_o3[:top_n]
    fronts = tools.sortNondominated(population, k=len(population), first_front_only=False)
    fronts = fronts[:n_fronts]

    def render_individual(ax, ind, img_chw, pred_c, pred_p, p_true):
        vis = alibi_chw_to_hwc_01(img_chw)
        ax.imshow(vis, vmin=0.0, vmax=1.0)
        ax.axis("off")
        f = ind.fitness.values
        txt = (
            f"pred={pred_c}  p(pred)={pred_p:.3f}\n"
            f"true={label_true}  p(true)={p_true:.3f}\n"
            f"O1={f[0]:.4f}  O2={f[1]:.2f}  O3={f[2]:.4f}"
        )
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.80, edgecolor="black", linewidth=0.3),
        )

    def page_for_individuals(pdf, title, inds, cols=3):
        if len(inds) == 0:
            return
        phenos = [ind.phenotype for ind in inds]
        imgs_np = process_phenotypes_for_images_rgb(
            phenos, original_img_np, turtle_delta_scale=turtle_delta_scale
        )
        pred_cls, pred_prob, probs = model_report_batch(
            model, imgs_np, device=device, batch_size=batch_size, amp=amp
        )
        K = len(inds)
        cols = min(cols, K)
        rows = int(math.ceil(K / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0), dpi=200)
        axes = np.array(axes).reshape(-1)
        for j in range(rows * cols):
            ax = axes[j]
            ax.axis("off")
            if j >= K:
                continue
            p_true = float(probs[j, label_true])
            render_individual(ax, inds[j], imgs_np[j], int(pred_cls[j]), float(pred_prob[j]), p_true)
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(wspace=0.35, hspace=0.35)
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(out_pdf_path) as pdf:
        page_for_individuals(pdf, f"Top-{top_n} by Objective 1 (min O1)", top_o1, cols=3)
        page_for_individuals(pdf, f"Top-{top_n} by Objective 2 (min O2)", top_o2, cols=3)
        page_for_individuals(pdf, f"Top-{top_n} by Objective 3 (min O3)", top_o3, cols=3)
        for fi, front in enumerate(fronts, start=1):
            page_for_individuals(pdf, f"Pareto Front {fi} (size={len(front)})", front, cols=ncols)


def save_final_population_pickle(
    population,
    original_img_np,
    out_pickle_path,
    label_true,
    device="cpu",
    batch_size=32,
    amp=False,
    turtle_delta_scale: float = 5.0,
    model_ckpt: str | None = None,
):
    ckpt = model_ckpt or str(_GRACO_ROOT / "models" / "imagenet100_inception.pt")
    model = load_task_classifier(ckpt, device)
    phenos = [ind.phenotype for ind in population]
    imgs_np = process_phenotypes_for_images_rgb(
        phenos, original_img_np, turtle_delta_scale=turtle_delta_scale
    )
    pred_cls_arr, pred_prob_arr, probs_arr = model_report_batch(
        model, imgs_np, device=device, batch_size=batch_size, amp=amp
    )
    n_cls = int(probs_arr.shape[1])
    sparsity_list = evaluate_sparsity_rgb(imgs_np, original_img_np)
    proximity_list = evaluate_proximity_rgb(imgs_np, original_img_np)
    data_list = []
    for idx, ind in enumerate(population):
        pred_c = int(pred_cls_arr[idx])
        pred_p = float(pred_prob_arr[idx])
        p_true = float(probs_arr[idx, label_true])
        o1_score = (
            p_true - float(probs_arr[idx].max())
            if pred_c != label_true
            else p_true - float(np.partition(probs_arr[idx], -2)[-2])
        )
        entry = {
            "ind_index": idx,
            "phenotype": ind.phenotype,
            "image": imgs_np[idx],
            "pred_class": pred_c,
            "pred_prob": pred_p,
            "p_true_class": p_true,
            "o1_score": o1_score,
            "sparsity_pixels": sparsity_list[idx],
            "proximity_l2": proximity_list[idx],
        }
        for c in range(n_cls):
            entry[f"p_class_{c}"] = float(probs_arr[idx, c])
        data_list.append(entry)
    with open(out_pickle_path, "wb") as f:
        pickle.dump(data_list, f)
    print(f"[save_final_population_pickle] saved {len(population)} individuals → {out_pickle_path}")


_TURTLE_DX = (1, 1, 0, -1, -1, -1, 0, 1)
_TURTLE_DY = (0, 1, 1, 1, 0, -1, -1, -1)


def _max_feasible_dist(cx: int, cy: int, d_idx: int, x_hi: int, y_hi: int) -> int:
    ddx = _TURTLE_DX[d_idx]
    ddy = _TURTLE_DY[d_idx]
    cands: list[int] = []
    if ddx > 0:
        cands.append((x_hi - cx) // ddx)
    elif ddx < 0:
        cands.append(cx // (-ddx))
    if ddy > 0:
        cands.append((y_hi - cy) // ddy)
    elif ddy < 0:
        cands.append(cy // (-ddy))
    if not cands:
        return 0
    return max(0, min(cands))


def find_choice(choice_list, genome, idx_genome):
    if len(choice_list) == 0:
        raise ValueError("Cannot choose from empty choice list")
    choice = genome[idx_genome] % len(choice_list)
    replacement = choice_list[choice]
    idx_genome = idx_genome + 1
    return replacement, idx_genome, choice, len(choice_list)


def find_choice_sensible(choice_list):
    if len(choice_list) == 0:
        raise ValueError("Cannot choose from empty choice list")
    choice = random.randint(0, len(choice_list) - 1)
    replacement = choice_list[choice]
    return replacement, choice, len(choice_list)


def find_replacement_turtle(x_max_arg, y_max_arg, genome, idx_genome_temp):
    x_hi, y_hi = x_max_arg, y_max_arg
    remainders, possible_choices = [], []
    x, idx_genome_temp, remainder, possible_choice = find_choice(
        list(range(0, x_max_arg + 1)), genome, idx_genome_temp
    )
    remainders.append(remainder)
    possible_choices.append(possible_choice)
    y, idx_genome_temp, remainder, possible_choice = find_choice(
        list(range(0, y_max_arg + 1)), genome, idx_genome_temp
    )
    remainders.append(remainder)
    possible_choices.append(possible_choice)
    num_steps, idx_genome_temp, remainder, possible_choice = find_choice(
        list(range(2, 8)), genome, idx_genome_temp
    )
    remainders.append(remainder)
    possible_choices.append(possible_choice)
    cx, cy = int(x), int(y)
    paths = []
    for _ in range(num_steps):
        d, idx_genome_temp, remainder, possible_choice = find_choice(
            list(range(8)), genome, idx_genome_temp
        )
        remainders.append(remainder)
        possible_choices.append(possible_choice)
        dmax = _max_feasible_dist(cx, cy, int(d), x_hi, y_hi)
        dist_choices = list(range(1, min(10, dmax) + 1)) if dmax >= 1 else [0]
        dist, idx_genome_temp, remainder, possible_choice = find_choice(
            dist_choices, genome, idx_genome_temp
        )
        remainders.append(remainder)
        possible_choices.append(possible_choice)
        paths.append(f"{d}:{dist}")
        cx += _TURTLE_DX[int(d)] * int(dist)
        cy += _TURTLE_DY[int(d)] * int(dist)
    return f'X-{x}-Y-{y}-Path-{",".join(paths)}', idx_genome_temp, remainders, possible_choices


def find_replacement_turtle_sensible(x_max_arg, y_max_arg):
    x_hi, y_hi = x_max_arg, y_max_arg
    remainders, possible_choices = [], []
    x, remainder, possible_choice = find_choice_sensible(list(range(0, x_max_arg + 1)))
    remainders.append(remainder)
    possible_choices.append(possible_choice)
    y, remainder, possible_choice = find_choice_sensible(list(range(0, y_max_arg + 1)))
    remainders.append(remainder)
    possible_choices.append(possible_choice)
    num_steps, remainder, possible_choice = find_choice_sensible(list(range(2, 8)))
    remainders.append(remainder)
    possible_choices.append(possible_choice)
    cx, cy = int(x), int(y)
    paths = []
    for _ in range(num_steps):
        d, remainder, possible_choice = find_choice_sensible(list(range(8)))
        remainders.append(remainder)
        possible_choices.append(possible_choice)
        dmax = _max_feasible_dist(cx, cy, int(d), x_hi, y_hi)
        dist_choices = list(range(1, min(10, dmax) + 1)) if dmax >= 1 else [0]
        dist, remainder, possible_choice = find_choice_sensible(dist_choices)
        remainders.append(remainder)
        possible_choices.append(possible_choice)
        paths.append(f"{d}:{dist}")
        cx += _TURTLE_DX[int(d)] * int(dist)
        cy += _TURTLE_DY[int(d)] * int(dist)
    return f'X-{x}-Y-{y}-Path-{",".join(paths)}', remainders, possible_choices


def decideTurtle(phenotype, idx_genome, genome):
    x_hi = x_max - 1
    y_hi = y_max - 1
    if idx_genome is not None:
        rep, idx_out, rems, pcs = find_replacement_turtle(x_hi, y_hi, genome, copy.deepcopy(idx_genome))
        return rep, idx_out - idx_genome, rems, pcs
    rep, rems, pcs = find_replacement_turtle_sensible(x_hi, y_hi)
    return rep, 0, rems, pcs


def evaluate_probability_inception(individuals_np, label, model, batch_size=32, device="cpu", amp=False):
    if isinstance(individuals_np, list):
        individuals_np = np.stack(individuals_np, axis=0)
    _, _, probs = predict_probs_numpy_inception(
        model, individuals_np, device=device, batch_size=batch_size, amp=amp
    )
    p_true = probs[:, label]
    probs_copy = probs.copy()
    probs_copy[:, label] = -np.inf
    p_other_max = np.max(probs_copy, axis=1)
    return (p_true - p_other_max).tolist()


def evaluate_sparsity_rgb(individuals_np, image, eps=1e-6, return_fraction=False):
    """Count spatial positions (H,W) where any RGB channel differs from the original"""
    if isinstance(individuals_np, list):
        individuals_np = np.stack(individuals_np, axis=0)
    changed = np.abs(individuals_np - image[np.newaxis, ...]) > eps
    any_ch = np.any(changed, axis=1)
    counts = np.count_nonzero(any_ch, axis=(1, 2))
    if return_fraction:
        _, h, w = image.shape
        return (counts / float(h * w)).tolist()
    return counts.tolist()


def evaluate_proximity_rgb(individuals_np, image):
    if isinstance(individuals_np, list):
        individuals_np = np.stack(individuals_np, axis=0)
    diff = individuals_np - image[np.newaxis, ...]
    return np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1).tolist()


def fitness_eval_parallel(
    individuals,
    points,
    image,
    label,
    model,
    batch_size,
    device,
    amp=False,
    turtle_delta_scale: float = 5.0,
):
    individuals_to_evaluate = []
    for ind in individuals:
        if not ind.fitness.valid:
            individuals_to_evaluate.append(ind.phenotype)
    individuals_np = process_phenotypes_for_images_rgb(
        individuals_to_evaluate, image, turtle_delta_scale=turtle_delta_scale
    )
    O_1_list = evaluate_probability_inception(
        individuals_np, label, model, batch_size, device, amp=amp
    )
    O_2_list = evaluate_sparsity_rgb(individuals_np, image)
    O_3_list = evaluate_proximity_rgb(individuals_np, image)
    O_1, O_2, O_3 = [], [], []
    counter = 0
    for ind in individuals:
        if ind.fitness.valid:
            O_1.append(1000)
            O_2.append(1000)
            O_3.append(1000)
            counter += 1
        else:
            o1_score = O_1_list[counter]
            O_1.append(o1_score)
            if o1_score >= 0:
                O_2.append(1000000)
                O_3.append(1000000)
            else:
                O_2.append(O_2_list[counter])
                O_3.append(O_3_list[counter])
            counter += 1
    return O_1, O_2, O_3


def evolveImage(config: dict, function_dict: dict, image: np.ndarray, label: int):
    defaults = {
        "population_size": 100,
        "max_generations": 50,
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
        "max_tree_depth": 35,
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
            "out_of_weight_indi",
        ],
        "n_runs": 1,
        "results_dir": "./results",
    }
    cfg = {**defaults, **(config or {})}
    print(cfg)

    toolbox = base.Toolbox()
    if not hasattr(creator, "FitnessMinIV3"):
        creator.create("FitnessMinIV3", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, "IndividualIV3"):
        creator.create("IndividualIV3", grape.Individual, fitness=creator.FitnessMinIV3)

    toolbox.register("populationCreator", grape.sensible_initialisation, creator.IndividualIV3)

    device = cfg.get("device", "cpu")
    batch_size_eval = int(cfg.get("batch_size_eval", 32))
    amp = bool(cfg.get("amp", False))
    turtle_delta_scale = float(cfg.get("turtle_delta_scale", 5.0))
    model_ckpt = _resolve_model_ckpt(cfg)

    model = load_task_classifier(model_ckpt, device)

    toolbox.register(
        "evaluate",
        fitness_eval_parallel,
        image=image,
        label=label,
        model=model,
        batch_size=batch_size_eval,
        device=device,
        amp=amp,
        turtle_delta_scale=turtle_delta_scale,
    )
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", grape.crossover_onepoint)
    toolbox.register("mutate", grape.mutation_int_flip_per_codon)

    collect_best = []
    hof_last = None
    for i in range(cfg["n_runs"]):
        seed = evolution_seed(cfg, i)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if cfg["random_initilisation"]:
            population = toolbox.populationCreator(
                pop_size=cfg["population_size"],
                bnf_grammar=BNF_GRAMMAR,
                min_init_genome_length=cfg["min_init_genome_length"],
                max_init_genome_length=cfg["max_init_genome_length"],
                max_init_depth=cfg["max_tree_depth"],
                codon_size=cfg["codon_size"],
                codon_consumption=cfg["codon_consumption"],
                genome_representation=cfg["genome_representation"],
            )
        else:
            population = toolbox.populationCreator(
                pop_size=cfg["population_size"],
                bnf_grammar=BNF_GRAMMAR,
                min_init_depth=cfg["min_init_tree_depth"],
                max_init_depth=cfg["max_init_tree_depth"],
                codon_size=cfg["codon_size"],
                codon_consumption=cfg["codon_consumption"],
                genome_representation=cfg["genome_representation"],
                function_dict=function_dict,
            )

        hof = tools.HallOfFame(cfg["halloffame_size"])
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.nanmean)
        stats.register("std", np.nanstd)
        stats.register("min", np.nanmin)
        stats.register("max", np.nanmax)
        register_objective_stats(stats)

        population, logbook = algorithms.ge_eaSimpleWithElitism(
            population,
            toolbox,
            cxpb=cfg["p_crossover"],
            mutpb=cfg["p_mutation"],
            ngen=cfg["max_generations"],
            elite_size=cfg["elite_size"],
            bnf_grammar=BNF_GRAMMAR,
            codon_size=cfg["codon_size"],
            max_tree_depth=cfg["max_tree_depth"],
            max_genome_length=cfg["max_genome_length"],
            points_train=None,
            points_test=None,
            codon_consumption=cfg["codon_consumption"],
            report_items=cfg["report_items"],
            genome_representation=cfg["genome_representation"],
            invalidate_max_depth=False,
            stats=stats,
            halloffame=hof,
            verbose=False,
            function_dict=function_dict,
        )

        os.makedirs(cfg["results_dir"], exist_ok=True)
        phenotypes_all = [ind.phenotype for ind in population]
        imgs_norm = process_phenotypes_for_images_rgb(
            phenotypes_all, image, turtle_delta_scale=turtle_delta_scale
        )
        pred_cls_arr, pred_prob_arr, probs_arr = model_report_batch(
            model, imgs_norm, device=device, batch_size=batch_size_eval, amp=amp
        )
        p_original_arr = probs_arr[:, int(label)]
        o1_arr = np.array([ind.fitness.values[0] for ind in population], dtype=np.float32)
        o2_arr = np.array([ind.fitness.values[1] for ind in population], dtype=np.float32)
        o3_arr = np.array([ind.fitness.values[2] for ind in population], dtype=np.float32)
        imgs_01 = np.stack([alibi_chw_to_01(imgs_norm[j]) for j in range(imgs_norm.shape[0])], axis=0)
        original_01 = alibi_chw_to_01(image)

        write_run_summary_npz(
            out_npz_path=cfg["results_dir"] + f"run_{i}_summary.npz",
            device=str(device),
            original_class=int(label),
            original_image_01=original_01,
            phenotypes=phenotypes_all,
            imgs_01=imgs_01,
            o1=o1_arr,
            o2=o2_arr,
            o3=o3_arr,
            pred_class=pred_cls_arr,
            pred_prob=pred_prob_arr,
            p_original=p_original_arr,
        )
        write_final_generation_csv(
            out_csv_path=cfg["results_dir"] + f"run_{i}_final_generation.csv",
            phenotypes=phenotypes_all,
            o1=o1_arr,
            o2=o2_arr,
            o3=o3_arr,
            original_class=int(label),
            p_original=p_original_arr,
            pred_class=pred_cls_arr,
            pred_prob=pred_prob_arr,
        )

        valid_population = [ind for ind in population if ind.fitness.values[0] < 0.50]
        if len(valid_population) > 5:
            save_detailed_report_pdf(
                population=valid_population,
                original_img_np=image,
                out_pdf_path=cfg["results_dir"] + f"run_{i}_report_imp.pdf",
                label_true=label,
                top_n=9,
                n_fronts=5,
                ncols=6,
                device=device,
                batch_size=batch_size_eval,
                amp=amp,
                turtle_delta_scale=turtle_delta_scale,
                model_ckpt=model_ckpt,
            )
        else:
            save_detailed_report_pdf(
                population=population,
                original_img_np=image,
                out_pdf_path=cfg["results_dir"] + f"run_{i}_report_not_imp.pdf",
                label_true=label,
                top_n=9,
                n_fronts=5,
                ncols=6,
                device=device,
                batch_size=batch_size_eval,
                amp=amp,
                turtle_delta_scale=turtle_delta_scale,
                model_ckpt=model_ckpt,
            )

        save_final_population_pickle(
            population=population,
            original_img_np=image,
            out_pickle_path=cfg["results_dir"] + f"run_{i}_final_pop.pkl",
            label_true=label,
            device=device,
            batch_size=batch_size_eval,
            amp=amp,
            turtle_delta_scale=turtle_delta_scale,
            model_ckpt=model_ckpt,
        )

        print("Best individual:\n", "\n".join(textwrap.wrap(hof.items[0].phenotype, 80)))
        collect_best.append(hof.items[0].fitness.values[0])
        hof_last = hof

        max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
        min_fitness_values, std_fitness_values = logbook.select("min", "std")
        best_ind_length = logbook.select("best_ind_length")
        avg_length = logbook.select("avg_length")
        selection_time = logbook.select("selection_time")
        generation_time = logbook.select("generation_time")
        gen, invalid = logbook.select("gen", "invalid")
        avg_used_codons = logbook.select("avg_used_codons")
        best_ind_used_codons = logbook.select("best_ind_used_codons")
        fitness_test = logbook.select("fitness_test")
        best_ind_nodes = logbook.select("best_ind_nodes")
        avg_nodes = logbook.select("avg_nodes")
        best_ind_depth = logbook.select("best_ind_depth")
        avg_depth = logbook.select("avg_depth")
        structural_diversity = logbook.select("structural_diversity")
        avg_obj = logbook.select("avg_obj")
        min_obj = logbook.select("min_obj")
        r = i
        header = list(GENERATION_LOG_COLUMNS)
        with open(cfg["results_dir"] + str(r) + "_Imagenet.csv", "w", encoding="UTF-8", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow(header)
            for value in range(len(max_fitness_values)):
                writer.writerow(
                    [
                        gen[value],
                        invalid[value],
                        mean_fitness_values[value],
                        std_fitness_values[value],
                        min_fitness_values[value],
                        max_fitness_values[value],
                        fitness_test[value],
                        best_ind_length[value],
                        avg_length[value],
                        best_ind_nodes[value],
                        avg_nodes[value],
                        best_ind_depth[value],
                        avg_depth[value],
                        avg_used_codons[value],
                        best_ind_used_codons[value],
                        structural_diversity[value],
                        selection_time[value],
                        generation_time[value],
                        *objective_row_suffix(avg_obj[value], min_obj[value]),
                    ]
                )
        write_objectives_by_generation(cfg["results_dir"], logbook)

    return collect_best, hof_last


def _parallel_effective(parallel_cfg: dict) -> dict:
    if not isinstance(parallel_cfg, dict):
        return {}
    return {k: v for k, v in parallel_cfg.items() if not str(k).startswith("_")}


def _job_worker(args: tuple):
    sample_index, run_id, label, image, job_config = args
    try:
        seed = evolution_seed({"run_id": run_id}, 0)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        fd = {"decideTurtle": decideTurtle}
        evolveImage(job_config, function_dict=fd, image=image, label=int(label))
        return (sample_index, run_id, "ok", None)
    except Exception as e:
        return (sample_index, run_id, "error", str(e))


def _load_config_top(path: str) -> dict:
    if load_config is not None:
        return load_config(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    print("Graco Inception v3 — RGB turtle, ImageNet val subset (100 classes)")

    parser = argparse.ArgumentParser(
        description="GracoImage Inception v3 + ImageNet val (see inception/README.md)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="JSON with runtime, paths, evolution, optional __parallel_jobs_inception__",
    )
    parser.add_argument("--parallel-config", type=str, default=None)
    parser.add_argument(
        "--imagenet-root",
        type=str,
        default=None,
        help="Override IMAGENET_ROOT (directory with train/ and val/)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="eval_manifest.json (list with dataset_index, y_true). "
        "If omitted, uses paths.eval_manifest from --config when that file exists.",
    )
    parser.add_argument("--test-run", action="store_true", help="1 sample, tiny evolution")
    parser.add_argument("--device", type=str, default=None, help="auto|cpu|cuda")
    args = parser.parse_args()

    parallel_cfg_raw: dict = {}
    cfg_top: dict = {}
    if args.parallel_config:
        pc_path = Path(args.parallel_config)
        pc_loaded = json.loads(pc_path.read_text(encoding="utf-8"))
        if not isinstance(pc_loaded, dict):
            raise ValueError("--parallel-config must be a JSON object")
        if "base_config" in pc_loaded:
            base_path = pc_loaded.get("base_config", "config.json")
            cfg_top = _load_config_top(str(_GRACO_ROOT / base_path))
            if not cfg_top:
                cfg_top = _load_config_top(base_path)
            parallel_cfg_raw = {k: v for k, v in pc_loaded.items() if k != "base_config"}
        else:
            cfg_top = _load_config_top(args.config)
            parallel_cfg_raw = pc_loaded
    else:
        cfg_top = _load_config_top(str(_GRACO_ROOT / args.config))
        if not cfg_top:
            cfg_top = _load_config_top(args.config)
        block = cfg_top.get("__parallel_jobs_inception__")
        parallel_cfg_raw = block if isinstance(block, dict) else {}

    parallel_cfg = _parallel_effective(parallel_cfg_raw)
    runtime = cfg_top.get("runtime", {}) if isinstance(cfg_top, dict) else {}
    paths = cfg_top.get("paths", {}) if isinstance(cfg_top, dict) else {}
    evo_cfg = cfg_top.get("evolution", {}) if isinstance(cfg_top, dict) else {}

    max_indices = int(parallel_cfg.get("max_indices", 100))
    runs_per_sample = int(parallel_cfg.get("runs_per_sample", 1))
    num_workers = int(parallel_cfg.get("num_workers", 1))
    experiment_name = parallel_cfg.get("experiment_name", "experiment_inception_turtle")
    test_run = bool(args.test_run or parallel_cfg.get("test_run", False))

    if test_run:
        max_indices = min(max_indices, 1)
        runs_per_sample = 1
    run_ids = [0] if test_run else scheduled_run_ids(parallel_cfg, runs_per_sample)

    device = args.device or runtime.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results_root = paths.get("results_root", "results")
    experiment_base_dir = os.path.join(_GRACO_ROOT, results_root, experiment_name, "")
    os.makedirs(experiment_base_dir, exist_ok=True)

    imagenet_root = (
        args.imagenet_root
        or paths.get("imagenet100_root")
        or paths.get("imagenet_root")
        or os.environ.get("IMAGENET_ROOT")
    )
    if imagenet_root:
        imagenet_root = str(Path(imagenet_root).expanduser())

    model_ckpt = paths.get("model_ckpt", "models/imagenet100_inception.pt")
    model_ckpt = str((_GRACO_ROOT / str(model_ckpt)).resolve())

    eval_manifest_path: str | None = None
    if args.manifest:
        manifest_p = Path(args.manifest).expanduser().resolve()
        if not manifest_p.is_file():
            raise SystemExit(f"--manifest not found: {manifest_p}")
        eval_manifest_path = str(manifest_p)
    elif paths.get("eval_manifest"):
        cand = (_GRACO_ROOT / str(paths["eval_manifest"])).resolve()
        if cand.is_file():
            eval_manifest_path = str(cand)

    # (results_folder_idx, y_true, img_chw) — folder_idx matches val ``dataset_index`` in manifest mode
    sample_rows: list[tuple[int, int, torch.Tensor]]

    if test_run:
        rng = np.random.default_rng(0)
        img = rng.uniform(low=-0.5, high=0.5, size=(3, 299, 299)).astype(np.float32)
        sample_rows = [(0, 0, torch.from_numpy(img))]
    else:
        n_classes = min(100, max(1, max_indices))

        if not eval_manifest_path:
            raise SystemExit(
                "eval_manifest.json is required. Pass --manifest or set paths.eval_manifest in config."
            )
        root_m = imagenet_root or (str(default_imagenet_root()) if default_imagenet_root() else None)
        if not root_m:
            raise SystemExit(
                "Set IMAGENET_ROOT, paths.imagenet100_root in config, or pass --imagenet-root."
            )
        entries = json.loads(Path(eval_manifest_path).read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise SystemExit(f"Manifest must be a JSON list: {eval_manifest_path}")
        rows = load_imagenet_val_manifest_samples(
            root_m,
            entries,
            max_samples=max_indices,
        )
        sample_rows = [(did, yt, torch.from_numpy(arr.copy())) for did, yt, arr in rows]
        print(f"Loaded {len(sample_rows)} sample(s) from manifest {eval_manifest_path}")

    jobs: list[tuple] = []
    n_sample = 0
    for folder_idx, y, img_t in sample_rows:
        if n_sample >= max_indices:
            break
        img_np = np.ascontiguousarray(img_t.numpy(), dtype=np.float32)
        for run_id in run_ids:
            job_cfg = dict(evo_cfg)
            job_cfg["n_runs"] = 1
            job_cfg["run_id"] = run_id
            job_cfg["max_generations"] = int(job_cfg.get("max_generations", 500))
            job_cfg["results_dir"] = os.path.join(
                experiment_base_dir, f"{folder_idx}_label_{y}", f"run_{run_id}", ""
            )
            job_cfg["device"] = device
            job_cfg["batch_size_eval"] = int(runtime.get("batch_size_eval", 32))
            job_cfg["amp"] = bool(runtime.get("amp", False))
            job_cfg["turtle_delta_scale"] = float(runtime.get("turtle_delta_scale", 5.0))
            job_cfg["model_ckpt"] = model_ckpt
            if test_run:
                job_cfg["population_size"] = 10
                job_cfg["max_generations"] = 2
            jobs.append((folder_idx, run_id, int(y), img_np.copy(), job_cfg))
        n_sample += 1

    total = len(jobs)
    if 0 in run_ids and not test_run:
        print("WARNING: run_id 0 is the finished seed-10 run and this job writes into run_0/.")
    print(
        f"Scheduled {total} job(s) num_workers={num_workers} "
        f"(max_indices={max_indices}, run_ids={run_ids})"
    )
    if total == 0:
        print("No jobs.")
    elif num_workers <= 1:
        for job in jobs:
            print(_job_worker(job))
    else:
        # spawn avoids fork-after-CUDA in the parent; workers load weights independently
        # without torch.hub multi-process lock contention.
        try:
            ctx = mp.get_context("spawn")
            executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
        except TypeError:
            executor = ProcessPoolExecutor(max_workers=num_workers)
        with executor as ex:
            futures = [ex.submit(_job_worker, job) for job in jobs]
            for fut in as_completed(futures):
                print(fut.result())
    print("Graco Inception finished.")
