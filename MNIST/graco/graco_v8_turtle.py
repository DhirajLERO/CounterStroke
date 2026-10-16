"""Boundary-aware turtle evolution on MNIST."""

import warnings


warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path

_GRACO_DIR = Path(__file__).resolve().parent
_MNIST_DIR = _GRACO_DIR.parent
_GRACOIMAGE_ROOT = _MNIST_DIR.parent
if str(_MNIST_DIR) not in sys.path:
    sys.path.insert(0, str(_MNIST_DIR))
if str(_GRACOIMAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRACOIMAGE_ROOT))

from explainers.mnist_cnn import MNISTNet  # noqa: E402

# Keep Matplotlib cache inside the repo (some environments disallow writing to $HOME).
_mpl_dir = Path(__file__).resolve().parent / ".mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — prevents X11 BadAlloc on long runs

import grape_v2 as grape
import algorithms_AG as algorithms

from deap import creator, base, tools
import random
import matplotlib.pyplot as plt
import copy
import re
import textwrap
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
from torchvision.utils import save_image


import math
import numpy as np
from common.objective_generation_log import (  # noqa: E402
    GENERATION_LOG_COLUMNS,
    evolution_seed,
    objective_row_suffix,
    register_objective_stats,
    scheduled_run_ids,
    write_objectives_by_generation,
)
from matplotlib.backends.backend_pdf import PdfPages
from deap import tools
import csv
import json
import pickle
try:
    import pandas as pd
except ImportError:
    pd = None

from concurrent.futures import ProcessPoolExecutor, as_completed

from config_loader import load_config
from infer_utils import predict_probs_numpy
from mnist_transforms import alibi_mnist_transform_hw, alibi_to_01


def resolve_model_ckpt(model_ckpt: str) -> str:
    """Resolve checkpoint relative to MNIST/ or GracoImage/."""
    p = Path(model_ckpt)
    if p.is_absolute():
        return str(p)
    graco = Path(__file__).resolve().parent
    for base in (_MNIST_DIR, _GRACOIMAGE_ROOT, graco):
        cand = (base / model_ckpt).resolve()
        if cand.is_file():
            return str(cand)
    return str((_MNIST_DIR / model_ckpt).resolve())


def _min_index_where(mask: np.ndarray, values: np.ndarray) -> int | None:
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return None
    return int(idxs[np.argmin(values[idxs])])


def _write_final_generation_csv(
    *,
    out_csv_path: str,
    phenotypes: list[str],
    o1: np.ndarray,
    o2: np.ndarray,
    o3: np.ndarray,
    original_class: int,
    p_original: np.ndarray,
    pred_class: np.ndarray,
    pred_prob: np.ndarray,
) -> None:
    header = [
        "ind_index",
        "phenotype",
        "o1",
        "o2",
        "o3",
        "original_class",
        "p_original_class",
        "pred_class",
        "pred_prob",
    ]

    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, ph in enumerate(phenotypes):
            w.writerow(
                [
                    i,
                    ph,
                    float(o1[i]),
                    float(o2[i]),
                    float(o3[i]),
                    int(original_class),
                    float(p_original[i]),
                    int(pred_class[i]),
                    float(pred_prob[i]),
                ]
            )


def _write_run_summary_npz(
    *,
    out_npz_path: str,
    device: str,
    original_class: int,
    original_image_01: np.ndarray,
    phenotypes: list[str],
    imgs_01: np.ndarray,
    o1: np.ndarray,
    o2: np.ndarray,
    o3: np.ndarray,
    pred_class: np.ndarray,
    pred_prob: np.ndarray,
    p_original: np.ndarray,
) -> None:
    # Selection indices
    idx_best_o1 = int(np.argmin(o1))
    idx_best_o2 = int(np.argmin(o2))
    idx_best_o3 = int(np.argmin(o3))

    mask_cf = pred_class != int(original_class)
    mask_cf_p05 = mask_cf & (pred_prob > 0.5)

    idx_cf_min_o2 = _min_index_where(mask_cf, o2)
    idx_cf_min_o3 = _min_index_where(mask_cf, o3)
    idx_cf_p05_min_o2 = _min_index_where(mask_cf_p05, o2)
    idx_cf_p05_min_o3 = _min_index_where(mask_cf_p05, o3)

    def pack(idx: int | None, prefix: str) -> dict:
        if idx is None:
            return {
                f"has_{prefix}": np.array(False),
                f"{prefix}_image": np.array([], dtype=np.float32),
                f"{prefix}_phenotype": np.array("", dtype=object),
                f"{prefix}_o1": np.array(np.nan, dtype=np.float32),
                f"{prefix}_o2": np.array(np.nan, dtype=np.float32),
                f"{prefix}_o3": np.array(np.nan, dtype=np.float32),
                f"{prefix}_pred_class": np.array(-1, dtype=np.int64),
                f"{prefix}_pred_prob": np.array(np.nan, dtype=np.float32),
                f"{prefix}_p_original": np.array(np.nan, dtype=np.float32),
            }

        return {
            f"has_{prefix}": np.array(True),
            f"{prefix}_image": imgs_01[idx].astype(np.float32, copy=False),
            f"{prefix}_phenotype": np.array(phenotypes[idx], dtype=object),
            f"{prefix}_o1": np.array(float(o1[idx]), dtype=np.float32),
            f"{prefix}_o2": np.array(float(o2[idx]), dtype=np.float32),
            f"{prefix}_o3": np.array(float(o3[idx]), dtype=np.float32),
            f"{prefix}_pred_class": np.array(int(pred_class[idx]), dtype=np.int64),
            f"{prefix}_pred_prob": np.array(float(pred_prob[idx]), dtype=np.float32),
            f"{prefix}_p_original": np.array(float(p_original[idx]), dtype=np.float32),
        }

    payload: dict = {
        "device": np.array(str(device), dtype=object),
        "original_class": np.array(int(original_class), dtype=np.int64),
        "original_image": original_image_01.astype(np.float32, copy=False),
    }

    payload.update(pack(idx_best_o1, "best_o1"))
    payload.update(pack(idx_best_o2, "best_o2"))
    payload.update(pack(idx_best_o3, "best_o3"))
    payload.update(pack(idx_cf_min_o2, "cf_min_o2"))
    payload.update(pack(idx_cf_min_o3, "cf_min_o3"))
    payload.update(pack(idx_cf_p05_min_o2, "cf_p05_min_o2"))
    payload.update(pack(idx_cf_p05_min_o3, "cf_p05_min_o3"))

    np.savez_compressed(out_npz_path, **payload)


problem = 'GracoImage'



x_max = 28
y_max = 28

GRAMMAR_FILE = "AG_turtle.bnf"
# GRAMMAR_FILE = "AG_rotate_one_patch.bnf"
BNF_GRAMMAR = grape.Grammar(r"grammar/" + GRAMMAR_FILE)

import numpy as np




def model_report_batch(model, imgs_np, device="cpu", batch_size=1024, amp=False):
    return predict_probs_numpy(model, imgs_np, device=device, batch_size=batch_size, amp=amp)

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from deap import tools
import torch



def _vis_01_from_alibi(img_alibi_2d):
    return alibi_to_01(img_alibi_2d)

def save_detailed_report_pdf(population, original_img_np, out_pdf_path,
                             model_path, label_true,
                             top_n=5,
                             n_fronts=3, ncols=8, device='cpu', batch_size=1024, amp=False,
                             turtle_delta_scale: float = 5.0):

    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pop_sorted_o1 = sorted(population, key=lambda ind: ind.fitness.values[0])
    pop_sorted_o2 = sorted(population, key=lambda ind: ind.fitness.values[1])
    pop_sorted_o3 = sorted(population, key=lambda ind: ind.fitness.values[2])

    top_o1 = pop_sorted_o1[:top_n]
    top_o2 = pop_sorted_o2[:top_n]
    top_o3 = pop_sorted_o3[:top_n]

    fronts = tools.sortNondominated(population, k=len(population), first_front_only=False)
    fronts = fronts[:n_fronts]

    def render_individual(ax, ind, img_norm_2d, pred_c, pred_p, p_true):
        ax.imshow(_vis_01_from_alibi(img_norm_2d), cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
        f = ind.fitness.values
        txt = (
            f"pred={pred_c}  p(pred)={pred_p:.3f}\n"
            f"true={label_true}  p(true)={p_true:.3f}\n"
            f"O1={f[0]:.4f}  O2={f[1]:.2f}  O3={f[2]:.4f}"
        )
        ax.text(
            0.02, 0.98, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=7,
            bbox=dict(facecolor="white", alpha=0.80, edgecolor="black", linewidth=0.3)
        )

    def page_for_individuals(pdf, title, inds, cols=3):
        if len(inds) == 0:
            return

        phenos = [ind.phenotype for ind in inds]
        imgs_np = process_phenotypes_for_images(
            phenos, original_img_np, turtle_delta_scale=turtle_delta_scale
        )  # (K,28,28) Alibi

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
            render_individual(ax, inds[j], imgs_np[j],
                              int(pred_cls[j]), float(pred_prob[j]), p_true)

        fig.suptitle(title, fontsize=14)
        # add separation
        fig.subplots_adjust(wspace=0.35, hspace=0.35)
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(out_pdf_path) as pdf:
        page_for_individuals(pdf, f"Top-{top_n} by Objective 1 (min O1)", top_o1, cols=3)
        page_for_individuals(pdf, f"Top-{top_n} by Objective 2 (min O2)", top_o2, cols=3)
        page_for_individuals(pdf, f"Top-{top_n} by Objective 3 (min O3)", top_o3, cols=3)

        for fi, front in enumerate(fronts, start=1):
            page_for_individuals(pdf, f"Pareto Front {fi} (size={len(front)})", front, cols=ncols)


def save_final_population_pickle(population, original_img_np, out_pickle_path,
                                 model_path, label_true, device='cpu', batch_size=1024, amp=False,
                                 turtle_delta_scale: float = 5.0):
    """Saves every individual in `population` to a pickle file with metrics"""
    import os
    import numpy as np
    import torch

    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    phenos = [ind.phenotype for ind in population]
    imgs_np = process_phenotypes_for_images(
        phenos, original_img_np, turtle_delta_scale=turtle_delta_scale
    )  # (N,28,28) Alibi

    pred_cls_arr, pred_prob_arr, probs_arr = model_report_batch(
        model, imgs_np, device=device, batch_size=batch_size, amp=amp
    )

    sparsity_list  = evaluate_sparsity(imgs_np, original_img_np)
    proximity_list = evaluate_proximity(imgs_np, original_img_np)

    data_list = []
    for idx, ind in enumerate(population):
        pred_c  = int(pred_cls_arr[idx])
        pred_p  = float(pred_prob_arr[idx])
        p_true  = float(probs_arr[idx, label_true])

        # Calculate O1 score (same as fitness_eval_parallel)
        o1_score = p_true - float(probs_arr[idx].max()) if pred_c != label_true else p_true - float(np.partition(probs_arr[idx], -2)[-2])

        entry = {
            "ind_index": idx,
            "phenotype": ind.phenotype,
            "image": imgs_np[idx],
            "pred_class": pred_c,
            "pred_prob": pred_p,
            "p_true_class": p_true,
            "o1_score": o1_score,
            "sparsity_pixels": sparsity_list[idx],
            "proximity_l2": proximity_list[idx]
        }
        # Add all per-class probabilities
        for c in range(10):
            entry[f"p_class_{c}"] = float(probs_arr[idx, c])

        data_list.append(entry)

    # Save as pickle
    with open(out_pickle_path, 'wb') as f:
        pickle.dump(data_list, f)

    print(f"[save_final_population_pickle] saved {len(population)} individuals → {out_pickle_path}")









def apply_turtle_to_image(img, xc, yc, path_str, sign, pct, fill=True, delta_scale: float = 5.0):
    import numpy as np
    from matplotlib.path import Path
    points = [(xc, yc)]
    curr_x, curr_y = xc, yc
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    for step in path_str.split(','):
        if not step: continue
        d_idx, dist = map(int, step.split(':'))
        curr_x += dx[d_idx] * dist
        curr_y += dy[d_idx] * dist
        points.append((curr_x, curr_y))

    lo, hi = -0.5, 0.5
    change_val = pct / 100.0
    step = change_val * float(delta_scale)
    h, w = img.shape

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
                    ref_pt = proj_pt + (proj_pt - pt)
                    mirrored.append(ref_pt)
            points.extend(mirrored)
        polygon = np.array(points)
        y_grid, x_grid = np.mgrid[:h, :w]
        points_grid = np.vstack((x_grid.flatten(), y_grid.flatten())).T
        if len(polygon) >= 3:
            p = Path(polygon)
            mask = p.contains_points(points_grid).reshape(h, w)
            if sign == "P": img[mask] += step
            else: img[mask] -= step
            return np.clip(img, lo, hi)
    else:
        mask = np.zeros((h, w), dtype=bool)
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            # Bresenham-style: sample along the segment
            num = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0)), 1) + 1
            xs = np.round(np.linspace(x0, x1, num)).astype(int)
            ys = np.round(np.linspace(y0, y1, num)).astype(int)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            mask[ys[valid], xs[valid]] = True
        if sign == "P": img[mask] += step
        else: img[mask] -= step
        return np.clip(img, lo, hi)

    return img

def apply_phenotype_to_image(phenotype, image, clip=None, turtle_delta_scale: float = 5.0):
    import numpy as np
    import re
    _SHAPE_RE = re.compile(r"^Shape-")
    # Matches:  X-<x>-Y-<y>-Path-<path>-Fill-<True|False>-Change-<P|N>-<dd.dd>
    _TURTLE_RE = re.compile(
        r"X-(\d+)-Y-(\d+)-Path-([\d:,]+)-Fill-(True|False)-Change-([PN])-(\d+\.\d+)"
    )
    out = image.copy()
    patches = [p for p in phenotype.split("__") if p]
    for patch in patches:
        mshape = _SHAPE_RE.match(patch)
        if not mshape: return out
        patch_body = patch[mshape.end():]
        for m in _TURTLE_RE.finditer(patch_body):
            fill = m.group(4) == "True"
            out = apply_turtle_to_image(
                out,
                int(m.group(1)), int(m.group(2)),
                m.group(3),
                m.group(5), float(m.group(6)),
                fill=fill,
                delta_scale=turtle_delta_scale,
            )
    if clip is not None:
        lo, hi = clip
        out = np.clip(out, lo, hi)
    return out

def process_phenotypes_for_images(individuals, image, clip=None, turtle_delta_scale: float = 5.0):
    phenos = []
    for ind in individuals:
        phenos.append(ind if isinstance(ind, str) else ind.phenotype)

    imgs = [
        apply_phenotype_to_image(p, image, clip=clip, turtle_delta_scale=turtle_delta_scale)
        for p in phenos
    ]
    return np.stack(imgs, axis=0)

def evaluate_probability(individuals_np, label, model, batch_size=256, device="cpu", amp=False):
    if isinstance(individuals_np, list):
        individuals_np = np.stack(individuals_np, axis=0)

    _, _, probs = predict_probs_numpy(
        model,
        individuals_np,
        device=device,
        batch_size=batch_size,
        amp=amp,
    )

    p_true = probs[:, label]
    probs_copy = probs.copy()
    probs_copy[:, label] = -np.inf
    p_other_max = np.max(probs_copy, axis=1)

    score = p_true - p_other_max
    return score.tolist()

import numpy as np

def evaluate_sparsity(individuals_np, image, eps=1e-6, return_fraction=False):

    if isinstance(individuals_np, list):
        individuals_np = np.stack(individuals_np, axis=0)  # (N,H,W)

    changed = np.abs(individuals_np - image[None, :, :]) > eps
    counts = np.count_nonzero(changed, axis=(1, 2))
    if return_fraction:
        h, w = image.shape
        return (counts / float(h * w)).tolist()
    return counts.tolist()


def evaluate_proximity(individuals_np, image):
    """
    Returns Euclidean distance per image to the original.
    """
    if isinstance(individuals_np, list):
        individuals_np = np.stack(individuals_np, axis=0)  # (N,H,W)

    diff = individuals_np - image[None, :, :]
    dists = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    return dists.tolist()


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


    individuals_np = process_phenotypes_for_images(
        individuals_to_evaluate, image, turtle_delta_scale=turtle_delta_scale
    )
    print(len(individuals_np))
    O_1_list = evaluate_probability(individuals_np, label, model, batch_size, device, amp=amp)
    O_2_list = evaluate_sparsity(individuals_np, image)

    O_3_list = evaluate_proximity(individuals_np, image)

    O_1 = []
    O_2 = []
    O_3 = []

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

            # Penalize O2 and O3 heavily if the model is not fooled yet (o1_score >= 0)
            if o1_score >= 0:
                O_2.append(1000000)
                O_3.append(1000000)
            else:
                O_2.append(O_2_list[counter])
                O_3.append(O_3_list[counter])

            # O_2.append(O_2_list[counter])
            # O_3.append(O_3_list[counter])




            counter += 1

    return O_1, O_2, O_3, # list of sparsity



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


# Same order as apply_turtle_to_image (direction index -> delta).
_TURTLE_DX = (1, 1, 0, -1, -1, -1, 0, 1)
_TURTLE_DY = (0, 1, 1, 1, 0, -1, -1, -1)


def _max_feasible_dist(cx: int, cy: int, d_idx: int, x_hi: int, y_hi: int) -> int:
    """Max integer t>=0 with (cx + dx*t, cy + dy*t) in [0,x_hi]×[0,y_hi]."""
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


def find_replacement_turtle(x_max, y_max, genome, idx_genome_temp):
    """x_max, y_max are max cell indices (27 for 28-wide MNIST), same as v7."""
    x_hi, y_hi = x_max, y_max
    remainders, possible_choices = [], []
    x, idx_genome_temp, remainder, possible_choice = find_choice(list(range(0, x_max + 1)), genome, idx_genome_temp)
    remainders.append(remainder); possible_choices.append(possible_choice)
    y, idx_genome_temp, remainder, possible_choice = find_choice(list(range(0, y_max + 1)), genome, idx_genome_temp)
    remainders.append(remainder); possible_choices.append(possible_choice)
    num_steps, idx_genome_temp, remainder, possible_choice = find_choice(list(range(2, 8)), genome, idx_genome_temp)
    remainders.append(remainder); possible_choices.append(possible_choice)
    cx, cy = int(x), int(y)
    paths = []
    for _ in range(num_steps):
        d, idx_genome_temp, remainder, possible_choice = find_choice(list(range(8)), genome, idx_genome_temp)
        remainders.append(remainder); possible_choices.append(possible_choice)
        dmax = _max_feasible_dist(cx, cy, int(d), x_hi, y_hi)
        if dmax >= 1:
            dist_choices = list(range(1, min(10, dmax) + 1))
        else:
            dist_choices = [0]
        dist, idx_genome_temp, remainder, possible_choice = find_choice(dist_choices, genome, idx_genome_temp)
        remainders.append(remainder); possible_choices.append(possible_choice)
        paths.append(f"{d}:{dist}")
        cx += _TURTLE_DX[int(d)] * int(dist)
        cy += _TURTLE_DY[int(d)] * int(dist)
    return f'X-{x}-Y-{y}-Path-{",".join(paths)}', idx_genome_temp, remainders, possible_choices


def find_replacement_turtle_sensible(x_max, y_max):
    x_hi, y_hi = x_max, y_max
    remainders, possible_choices = [], []
    x, remainder, possible_choice = find_choice_sensible(list(range(0, x_max + 1)))
    remainders.append(remainder); possible_choices.append(possible_choice)
    y, remainder, possible_choice = find_choice_sensible(list(range(0, y_max + 1)))
    remainders.append(remainder); possible_choices.append(possible_choice)
    num_steps, remainder, possible_choice = find_choice_sensible(list(range(2, 8)))
    remainders.append(remainder); possible_choices.append(possible_choice)
    cx, cy = int(x), int(y)
    paths = []
    for _ in range(num_steps):
        d, remainder, possible_choice = find_choice_sensible(list(range(8)))
        remainders.append(remainder); possible_choices.append(possible_choice)
        dmax = _max_feasible_dist(cx, cy, int(d), x_hi, y_hi)
        if dmax >= 1:
            dist_choices = list(range(1, min(10, dmax) + 1))
        else:
            dist_choices = [0]
        dist, remainder, possible_choice = find_choice_sensible(dist_choices)
        remainders.append(remainder); possible_choices.append(possible_choice)
        paths.append(f"{d}:{dist}")
        cx += _TURTLE_DX[int(d)] * int(dist)
        cy += _TURTLE_DY[int(d)] * int(dist)
    return f'X-{x}-Y-{y}-Path-{",".join(paths)}', remainders, possible_choices


def decideTurtle(phenotype, idx_genome, genome):
    x_hi = x_max - 1
    y_hi = y_max - 1
    if idx_genome is not None:
        import copy
        rep, idx_out, rems, pcs = find_replacement_turtle(x_hi, y_hi, genome, copy.deepcopy(idx_genome))
        return rep, idx_out - idx_genome, rems, pcs
    else:
        rep, rems, pcs = find_replacement_turtle_sensible(x_hi, y_hi)
        return rep, 0, rems, pcs

def evolveImage(config: dict, function_dict: dict, image, label):
    global xmax
    global ymax
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
            "gen", "invalid", "avg", "std", "min", "max",
            "fitness_test",
            "best_ind_length", "avg_length",
            "best_ind_nodes", "avg_nodes",
            "best_ind_depth", "avg_depth",
            "avg_used_codons", "best_ind_used_codons",
            "structural_diversity",
            "selection_time", "generation_time", "out_of_weight_indi",
        ],

        "n_runs": 1,
        "results_dir": "./results",
    }

    # merge: user-provided config overrides defaults
    cfg = {**defaults, **(config or {})}
    print(cfg)


    toolbox = base.Toolbox()

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", grape.Individual, fitness=creator.FitnessMin)

    toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
    # toolbox.register("evaluate", fitness_eval)

    model_ckpt = resolve_model_ckpt(str(cfg.get("model_ckpt", "Model_training/mnist_cnn.pt")))
    device = cfg.get("device", "cpu")
    batch_size_eval = int(cfg.get("batch_size_eval", 1000))
    amp = bool(cfg.get("amp", False))
    turtle_delta_scale = float(cfg.get("turtle_delta_scale", 5.0))

    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()


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
    # toolbox.register("select", tools.selTournament, tournsize=cfg["tournsize"])
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", grape.crossover_onepoint)
    toolbox.register("mutate", grape.mutation_int_flip_per_codon)

    collect_best = []
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
            population, toolbox,
            cxpb=cfg["p_crossover"], mutpb=cfg["p_mutation"],
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
            stats=stats, halloffame=hof, verbose=False,
            function_dict=function_dict
        )

        os.makedirs(cfg["results_dir"], exist_ok=True)

        phenotypes_all = [ind.phenotype for ind in population]
        imgs_norm = process_phenotypes_for_images(
            phenotypes_all, image, turtle_delta_scale=turtle_delta_scale
        )  # (N,28,28) Alibi
        pred_cls_arr, pred_prob_arr, probs_arr = model_report_batch(
            model, imgs_norm, device=device, batch_size=batch_size_eval, amp=amp
        )
        p_original_arr = probs_arr[:, int(label)]

        # objectives from final generation
        o1_arr = np.array([ind.fitness.values[0] for ind in population], dtype=np.float32)
        o2_arr = np.array([ind.fitness.values[1] for ind in population], dtype=np.float32)
        o3_arr = np.array([ind.fitness.values[2] for ind in population], dtype=np.float32)

        # [0,1] images for saving (Alibi -> physical grayscale)
        imgs_01 = np.stack([alibi_to_01(imgs_norm[j]) for j in range(imgs_norm.shape[0])], axis=0)
        original_01 = alibi_to_01(image)

        _write_run_summary_npz(
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

        _write_final_generation_csv(
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


        valid_population = [individual for individual in population if individual.fitness.values[0] < 0.50]

        if len(valid_population) > 5:
            save_detailed_report_pdf(
                population=valid_population,
                original_img_np=image,
                out_pdf_path=cfg["results_dir"] + f'run_{i}_report_imp.pdf',
                model_path=model_ckpt,
                label_true=label,
                top_n=9,
                n_fronts=5,
                ncols=6,
                device=device,
                batch_size=batch_size_eval,
                amp=amp,
                turtle_delta_scale=turtle_delta_scale,
            )
        else:
            save_detailed_report_pdf(
                population=population,
                original_img_np=image,
                out_pdf_path=cfg["results_dir"] + f'run_{i}_report_not_imp.pdf',
                model_path=model_ckpt,
                label_true=label,
                top_n=9,
                n_fronts=5,
                ncols=6,
                device=device,
                batch_size=batch_size_eval,
                amp=amp,
                turtle_delta_scale=turtle_delta_scale,
            )

        save_final_population_pickle(
            population=population,
            original_img_np=image,
            out_pickle_path=cfg["results_dir"] + f'run_{i}_final_pop.pkl',
            model_path=model_ckpt,
            label_true=label,
            device=device,
            batch_size=batch_size_eval,
            amp=amp,
            turtle_delta_scale=turtle_delta_scale,
        )

        best = hof.items[0].phenotype
        print(best)
        print("Best individual: \n", "\n".join(textwrap.wrap(best, 80)))
        print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
        collect_best.append(hof.items[0].fitness.values[0])

        print("Depth: ", hof.items[0].depth)
        print("Length of the genome: ", len(hof.items[0].genome))
        print(f'Used portion of the genome: {hof.items[0].used_codons / len(hof.items[0].genome):.2f}')

        collect_best.append(hof.items[0].fitness.values[0])

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

        with open(cfg["results_dir"] + str(r) + "_MNIST.csv", "w", encoding='UTF8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(header)
            for value in range(len(max_fitness_values)):
                writer.writerow([gen[value], invalid[value], mean_fitness_values[value],
                                 std_fitness_values[value], min_fitness_values[value],
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
                                 *objective_row_suffix(avg_obj[value], min_obj[value])])

        write_objectives_by_generation(cfg["results_dir"], logbook)


    return collect_best, hof


def load_dataset(label=5, train=False, data_dir="data", download=False):
    transform = alibi_mnist_transform_hw()
    dataset = datasets.MNIST(data_dir, train=train, download=download, transform=transform)

    return dataset


def _v8_parallel_effective(parallel_cfg: dict) -> dict:
    """Scheduling keys only; strip underscore-prefixed entries (e.g. _description)."""
    if not isinstance(parallel_cfg, dict):
        return {}
    return {k: v for k, v in parallel_cfg.items() if not str(k).startswith("_")}


def _v8_job_worker(args: tuple) -> tuple:
    """One evolution job: (sample_index, run_id, label, image_np, job_config_dict)."""
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


if "__main__" == __name__:
    print("GracoImage v8 (parallel sample×run jobs, boundary-aware turtle)")

    import argparse

    parser = argparse.ArgumentParser(
        description="GracoImage v8 turtle — parallel jobs; scheduling in __parallel_jobs_v8__ (see module docstring)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Main JSON config (runtime, paths, evolution, and optional __parallel_jobs_v8__)",
    )
    parser.add_argument(
        "--parallel-config",
        type=str,
        default=None,
        help="Optional: separate JSON for scheduling only, or legacy file with base_config + parallel fields",
    )
    parser.add_argument("--test-run", action="store_true", help="Smoke: 1 sample, 1 run, tiny evolution per job")
    parser.add_argument("--device", type=str, default=None, help="Override device for all jobs: auto|cpu|cuda")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="JSON file: list of {sample_idx, y_true} — run only these MNIST test indices; folders use sample_idx.",
    )
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
            cfg_top = load_config(base_path)
            parallel_cfg_raw = {k: v for k, v in pc_loaded.items() if k != "base_config"}
        else:
            cfg_top = load_config(args.config)
            parallel_cfg_raw = pc_loaded
    else:
        cfg_top = load_config(args.config)
        block = cfg_top.get("__parallel_jobs_v8__")
        parallel_cfg_raw = block if isinstance(block, dict) else {}

    parallel_cfg = _v8_parallel_effective(parallel_cfg_raw)
    runtime = cfg_top.get("runtime", {}) if isinstance(cfg_top, dict) else {}
    paths = cfg_top.get("paths", {}) if isinstance(cfg_top, dict) else {}
    evo_cfg = cfg_top.get("evolution", {}) if isinstance(cfg_top, dict) else {}

    max_indices = int(parallel_cfg.get("max_indices", 1))
    runs_per_sample = int(parallel_cfg.get("runs_per_sample", 1))
    num_workers = int(parallel_cfg.get("num_workers", 1))
    experiment_name = parallel_cfg.get("experiment_name", "experiment_v8_turtle")
    test_run = bool(args.test_run or parallel_cfg.get("test_run", False))

    if test_run:
        max_indices = min(max_indices, 1)
        runs_per_sample = 1
    run_ids = [0] if test_run else scheduled_run_ids(parallel_cfg, runs_per_sample)

    device = args.device or runtime.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results_root = paths.get("results_root", "results")
    experiment_base_dir = f"./{results_root}/{experiment_name}/"
    os.makedirs(experiment_base_dir, exist_ok=True)

    manifest_pairs: list[tuple[int, tuple]] | None = None
    raw_ds = None
    if test_run:
        rng = np.random.default_rng(0)
        img = rng.uniform(low=-0.5, high=0.5, size=(28, 28)).astype(np.float32)
        y = 0
        img_t = torch.from_numpy(img)
    else:
        data_dir = paths.get("data_dir", "data")
        download = bool(paths.get("download", True))
        raw_ds = load_dataset(data_dir=data_dir, download=download)
        if args.manifest:
            entries = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            manifest_pairs = [
                (int(e["sample_idx"]), raw_ds[int(e["sample_idx"])]) for e in entries
            ]

    jobs: list[tuple] = []
    if test_run:
        img_np = np.ascontiguousarray(img_t.squeeze().numpy(), dtype=np.float32)
        job_cfg = dict(evo_cfg)
        job_cfg["n_runs"] = 1
        job_cfg["run_id"] = 0
        job_cfg["max_generations"] = 2
        job_cfg["results_dir"] = f"{experiment_base_dir}0_label_{y}/run_0/"
        job_cfg["device"] = device
        job_cfg["batch_size_eval"] = runtime.get("batch_size_eval", 1000)
        job_cfg["amp"] = runtime.get("amp", False)
        job_cfg["model_ckpt"] = paths.get("model_ckpt", "models/mnist_cnn.pt")
        job_cfg["turtle_delta_scale"] = float(runtime.get("turtle_delta_scale", 5.0))
        job_cfg["population_size"] = 10
        jobs.append((0, 0, int(y), img_np.copy(), job_cfg))
    else:
        if manifest_pairs is not None:
            n_iter = min(max_indices, len(manifest_pairs))
        else:
            n_iter = min(max_indices, len(raw_ds))
        for loop_i in range(n_iter):
            if manifest_pairs is not None:
                folder_idx, (img_t, y) = manifest_pairs[loop_i]
            else:
                folder_idx = loop_i
                img_t, y = raw_ds[loop_i]
            img_np = np.ascontiguousarray(img_t.squeeze().numpy(), dtype=np.float32)
            for run_id in run_ids:
                job_cfg = dict(evo_cfg)
                job_cfg["n_runs"] = 1
                job_cfg["run_id"] = run_id
                job_cfg["max_generations"] = int(job_cfg.get("max_generations", 500))
                job_cfg["results_dir"] = f"{experiment_base_dir}{folder_idx}_label_{y}/run_{run_id}/"
                job_cfg["device"] = device
                job_cfg["batch_size_eval"] = runtime.get("batch_size_eval", 1000)
                job_cfg["amp"] = runtime.get("amp", False)
                job_cfg["model_ckpt"] = paths.get("model_ckpt", "models/mnist_cnn.pt")
                job_cfg["turtle_delta_scale"] = float(runtime.get("turtle_delta_scale", 5.0))

                jobs.append((folder_idx, run_id, int(y), img_np.copy(), job_cfg))

    total = len(jobs)
    if 0 in run_ids and not test_run:
        print("WARNING: run_id 0 is the finished seed-10 run and this job writes into run_0/.")
    print(
        f"Scheduled {total} job(s) with num_workers={num_workers} "
        f"(max_indices={max_indices}, run_ids={run_ids})"
    )

    if total == 0:
        print("No jobs to run.")
    elif num_workers <= 1:
        for job in jobs:
            r = _v8_job_worker(job)
            print(r)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_v8_job_worker, job) for job in jobs]
            for fut in as_completed(futures):
                print(fut.result())

    print("GracoImage v8 finished.")



