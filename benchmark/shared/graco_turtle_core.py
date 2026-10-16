"""Turtle evolution core for benchmark datasets (attribute-intervention fitness)."""

from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path

import numpy as np

_SHARED = Path(__file__).resolve().parent
_GRACO_LIBS = _SHARED / "graco_libs"
if str(_GRACO_LIBS) not in sys.path:
    sys.path.insert(0, str(_GRACO_LIBS))

import grape_v2 as grape  # noqa: E402
import algorithms_AG as algorithms  # noqa: E402
from config_loader import load_config  # noqa: E402

from deap import base, creator, tools  # noqa: E402

import torch  # noqa: E402

from graco_fitness import intervention_loss_celeba, intervention_loss_morphomnist  # noqa: E402
from graco_variant_pick import EvolvedCandidate  # noqa: E402

# Set by evolve_benchmark_image before grammar use
x_max: int = 32
y_max: int = 32
clip_lo: float = -1.0
clip_hi: float = 1.0
n_channels: int = 1

_TURTLE_DX = (1, 1, 0, -1, -1, -1, 0, 1)
_TURTLE_DY = (0, 1, 1, 1, 0, -1, -1, -1)


def apply_turtle_to_image(img, xc, yc, path_str, sign, pct, fill=True, delta_scale: float = 5.0):
    from matplotlib.path import Path as MplPath

    points = [(xc, yc)]
    curr_x, curr_y = xc, yc
    dx, dy = _TURTLE_DX, _TURTLE_DY
    for step in path_str.split(","):
        if not step:
            continue
        d_idx, dist = map(int, step.split(":"))
        curr_x += dx[d_idx] * dist
        curr_y += dy[d_idx] * dist
        points.append((curr_x, curr_y))

    change_val = pct / 100.0
    step_v = change_val * float(delta_scale)
    if img.ndim == 2:
        h, w = img.shape
        chans = [img]
    else:
        h, w, _ = img.shape
        chans = [img[:, :, c] for c in range(img.shape[2])]

    def _edit_plane(plane):
        if fill:
            start_pt = np.array(points[0], dtype=float)
            end_pt = np.array(points[-1], dtype=float)
            if np.linalg.norm(start_pt - end_pt) > 0.1:
                axis = end_pt - start_pt
                axis_len_sq = np.dot(axis, axis)
                mirrored = []
                for pt in reversed(points[1:-1]):
                    pt = np.array(pt, dtype=float)
                    if axis_len_sq < 1e-5:
                        mirrored.append(pt)
                    else:
                        v = pt - start_pt
                        proj_len = np.dot(v, axis) / axis_len_sq
                        proj_pt = start_pt + proj_len * axis
                        mirrored.append(proj_pt + (proj_pt - pt))
                points.extend(mirrored)
            polygon = np.array(points)
            y_grid, x_grid = np.mgrid[:h, :w]
            grid = np.vstack((x_grid.flatten(), y_grid.flatten())).T
            if len(polygon) >= 3:
                mask = MplPath(polygon).contains_points(grid).reshape(h, w)
                if sign == "P":
                    plane[mask] += step_v
                else:
                    plane[mask] -= step_v
        else:
            mask = np.zeros((h, w), dtype=bool)
            for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
                num = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0)), 1) + 1
                xs = np.round(np.linspace(x0, x1, num)).astype(int)
                ys = np.round(np.linspace(y0, y1, num)).astype(int)
                valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                mask[ys[valid], xs[valid]] = True
            if sign == "P":
                plane[mask] += step_v
            else:
                plane[mask] -= step_v
        return np.clip(plane, clip_lo, clip_hi)

    out = [_edit_plane(c.copy()) for c in chans]
    if len(out) == 1:
        return out[0]
    return np.stack(out, axis=-1)


_SHAPE_RE = re.compile(r"^Shape-")
# Matches each turtle in a patch body (multiple per Shape- via _<patch> chains).
_TURTLE_RE = re.compile(
    r"X-(\d+)-Y-(\d+)-Path-([\d:,]+)-Fill-(True|False)-Change-([PN])-(\d+\.\d+)"
)


def apply_phenotype_to_image(phenotype, image, turtle_delta_scale: float = 5.0):
    """Apply all turtle ops in phenotype (aligned with MNIST/CIFAR graco_v8_turtle)."""
    out = image.copy()
    for patch in [p for p in phenotype.split("__") if p]:
        mshape = _SHAPE_RE.match(patch)
        if not mshape:
            continue
        patch_body = patch[mshape.end() :]
        for m in _TURTLE_RE.finditer(patch_body):
            fill = m.group(4) == "True"
            out = apply_turtle_to_image(
                out,
                int(m.group(1)),
                int(m.group(2)),
                m.group(3),
                m.group(5),
                float(m.group(6)),
                fill=fill,
                delta_scale=turtle_delta_scale,
            )
    return out


def process_phenotypes_for_images(phenotypes, image, turtle_delta_scale=5.0):
    imgs = [apply_phenotype_to_image(p, image, turtle_delta_scale=turtle_delta_scale) for p in phenotypes]
    return np.stack(imgs, axis=0)


def evaluate_sparsity(individuals_np, image, eps=1e-6):
    if isinstance(individuals_np, list):
        individuals_np = np.stack(individuals_np, axis=0)
    if image.ndim == 2:
        changed = np.abs(individuals_np - image[None, :, :]) > eps
        return np.count_nonzero(changed, axis=(1, 2)).tolist()
    changed = np.abs(individuals_np - image[None, :, :, :]) > eps
    return np.count_nonzero(changed, axis=(1, 2, 3)).tolist()


def evaluate_proximity(individuals_np, image):
    if isinstance(individuals_np, list):
        individuals_np = np.stack(individuals_np, axis=0)
    diff = individuals_np - image[None, ...]
    return np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1).tolist()


def _numpy_to_nchw(batch_hw_or_hwc: np.ndarray) -> torch.Tensor:
    arr = batch_hw_or_hwc.astype(np.float32)
    if arr.ndim == 4 and arr.shape[-1] in (1, 3):
        # N,H,W,C -> N,C,H,W (CelebA RGB batch)
        t = torch.from_numpy(arr).permute(0, 3, 1, 2)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        # N,H,W,C (single RGB image wrongly batched as 3D — rare)
        t = torch.from_numpy(arr).permute(0, 3, 1, 2)
    else:
        # N,H,W grayscale (MorphoMNIST)
        t = torch.from_numpy(arr).unsqueeze(1)
    return t


def fitness_eval_attribute(
    individuals,
    _points,
    image,
    targets,
    predictors,
    dataset_name,
    batch_size,
    device,
    turtle_delta_scale=5.0,
):
    to_eval = [ind.phenotype for ind in individuals if not ind.fitness.valid]
    if not to_eval:
        return [], [], []

    individuals_np = process_phenotypes_for_images(to_eval, image, turtle_delta_scale=turtle_delta_scale)
    imgs_t = _numpy_to_nchw(individuals_np).to(device)

    if dataset_name == "morphomnist":
        o1_list = intervention_loss_morphomnist(predictors, imgs_t, targets, device=device).tolist()
    else:
        o1_list = intervention_loss_celeba(predictors, imgs_t, targets, device=device).tolist()

    o2_list = evaluate_sparsity(individuals_np, image)
    o3_list = evaluate_proximity(individuals_np, image)

    o1, o2, o3 = [], [], []
    j = 0
    for ind in individuals:
        if ind.fitness.valid:
            o1.append(1e6)
            o2.append(1e6)
            o3.append(1e6)
        else:
            o1.append(o1_list[j])
            o2.append(o2_list[j])
            o3.append(o3_list[j])
            j += 1
    return o1, o2, o3


def find_choice(choice_list, genome, idx_genome):
    choice = genome[idx_genome] % len(choice_list)
    return choice_list[choice], idx_genome + 1, choice, len(choice_list)


def find_choice_sensible(choice_list):
    choice = random.randint(0, len(choice_list) - 1)
    return choice_list[choice], choice, len(choice_list)


def _max_feasible_dist(cx, cy, d_idx, x_hi, y_hi):
    ddx, ddy = _TURTLE_DX[d_idx], _TURTLE_DY[d_idx]
    cands = []
    if ddx > 0:
        cands.append((x_hi - cx) // ddx)
    elif ddx < 0:
        cands.append(cx // (-ddx))
    if ddy > 0:
        cands.append((y_hi - cy) // ddy)
    elif ddy < 0:
        cands.append(cy // (-ddy))
    return max(0, min(cands)) if cands else 0


def find_replacement_turtle(x_hi, y_hi, genome, idx_genome_temp):
    remainders, possible_choices, paths = [], [], []
    x, idx_genome_temp, remainder, possible_choice = find_choice(list(range(0, x_hi + 1)), genome, idx_genome_temp)
    remainders.append(remainder)
    possible_choices.append(possible_choice)
    y, idx_genome_temp, remainder, possible_choice = find_choice(list(range(0, y_hi + 1)), genome, idx_genome_temp)
    remainders.append(remainder)
    possible_choices.append(possible_choice)
    num_steps, idx_genome_temp, remainder, possible_choice = find_choice(list(range(2, 8)), genome, idx_genome_temp)
    remainders.append(remainder)
    possible_choices.append(possible_choice)
    cx, cy = int(x), int(y)
    for _ in range(num_steps):
        d, idx_genome_temp, remainder, possible_choice = find_choice(list(range(8)), genome, idx_genome_temp)
        remainders.append(remainder)
        possible_choices.append(possible_choice)
        dmax = _max_feasible_dist(cx, cy, int(d), x_hi, y_hi)
        if dmax >= 1:
            dist_choices = list(range(1, min(20, dmax) + 1))
        else:
            dist_choices = [0]
        dist, idx_genome_temp, remainder, possible_choice = find_choice(dist_choices, genome, idx_genome_temp)
        remainders.append(remainder)
        possible_choices.append(possible_choice)
        paths.append(f"{d}:{dist}")
        cx += _TURTLE_DX[int(d)] * int(dist)
        cy += _TURTLE_DY[int(d)] * int(dist)
    return f"X-{x}-Y-{y}-Path-{','.join(paths)}", idx_genome_temp, remainders, possible_choices


def find_replacement_turtle_sensible(x_hi, y_hi):
    remainders, possible_choices = [], []
    x, remainder, possible_choice = find_choice_sensible(list(range(0, x_hi + 1)))
    remainders.append(remainder)
    possible_choices.append(possible_choice)
    y, remainder, possible_choice = find_choice_sensible(list(range(0, y_hi + 1)))
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
        if dmax >= 1:
            dist_choices = list(range(1, min(20, dmax) + 1))
        else:
            dist_choices = [0]
        dist, remainder, possible_choice = find_choice_sensible(dist_choices)
        remainders.append(remainder)
        possible_choices.append(possible_choice)
        paths.append(f"{d}:{dist}")
        cx += _TURTLE_DX[int(d)] * int(dist)
        cy += _TURTLE_DY[int(d)] * int(dist)
    return f"X-{x}-Y-{y}-Path-{','.join(paths)}", remainders, possible_choices


def decideTurtle(phenotype, idx_genome, genome):
    import copy

    x_hi, y_hi = x_max - 1, y_max - 1
    if idx_genome is not None:
        rep, idx_out, rems, pcs = find_replacement_turtle(x_hi, y_hi, genome, copy.deepcopy(idx_genome))
        return rep, idx_out - idx_genome, rems, pcs
    rep, rems, pcs = find_replacement_turtle_sensible(x_hi, y_hi)
    return rep, 0, rems, pcs


def evolve_benchmark_image(
    *,
    image_np: np.ndarray,
    targets: dict,
    predictors: dict,
    dataset_name: str,
    grammar_path: Path,
    config: dict,
    device: str = "cuda",
) -> list[EvolvedCandidate]:
    """Run one evolution; return candidate list with rendered images for variant picking."""
    global BNF_GRAMMAR

    BNF_GRAMMAR = grape.Grammar(str(grammar_path))
    function_dict = {"decideTurtle": decideTurtle}

    cfg = {
        "population_size": 80,
        "max_generations": 40,
        "p_crossover": 0.9,
        "p_mutation": 0.01,
        "elite_size": 1,
        "halloffame_size": 30,
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
        "n_runs": 1,
        "device": device,
        "batch_size_eval": 64,
        "turtle_delta_scale": 5.0,
        **(config or {}),
    }

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", grape.Individual, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
    toolbox.register(
        "evaluate",
        fitness_eval_attribute,
        image=image_np,
        targets=targets,
        predictors=predictors,
        dataset_name=dataset_name,
        batch_size=cfg["batch_size_eval"],
        device=device,
        turtle_delta_scale=cfg["turtle_delta_scale"],
    )
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", grape.crossover_onepoint)
    toolbox.register("mutate", grape.mutation_int_flip_per_codon)

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
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.nanmin)

    population, _ = algorithms.ge_eaSimpleWithElitism(
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
        report_items=["gen", "min"],
        genome_representation=cfg["genome_representation"],
        stats=stats,
        halloffame=hof,
        verbose=False,
        function_dict=function_dict,
    )

    seen = set()
    candidates: list[EvolvedCandidate] = []
    for ind in list(hof) + list(population):
        if not ind.fitness.valid:
            continue
        key = ind.phenotype
        if key in seen:
            continue
        seen.add(key)
        o1, o2, o3 = ind.fitness.values
        img = apply_phenotype_to_image(key, image_np, turtle_delta_scale=cfg["turtle_delta_scale"])
        candidates.append(EvolvedCandidate(phenotype=key, o1=float(o1), o2=float(o2), o3=float(o3), image_np=img))
    if not candidates:
        best = min(population, key=lambda ind: ind.fitness.values[0])
        img = apply_phenotype_to_image(best.phenotype, image_np, turtle_delta_scale=cfg["turtle_delta_scale"])
        o1, o2, o3 = best.fitness.values
        candidates.append(
            EvolvedCandidate(phenotype=best.phenotype, o1=float(o1), o2=float(o2), o3=float(o3), image_np=img)
        )
    return candidates
