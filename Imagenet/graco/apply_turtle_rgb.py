"""RGB turtle rendering: phenotype strings use Shape-<R|G|B>- then the same"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
from matplotlib.path import Path

Channel = Literal["R", "G", "B"]
_CHANNEL_TO_IDX = {"R": 0, "G": 1, "B": 2}


def apply_turtle_to_image_2d(
    img_hw: np.ndarray,
    xc: int,
    yc: int,
    path_str: str,
    sign: str,
    pct: float,
    fill: bool = True,
    delta_scale: float = 5.0,
) -> np.ndarray:
    """Mutate single-channel 2D array (H,W), same geometry as v8 ``apply_turtle_to_image``."""
    img = img_hw
    points = [(xc, yc)]
    curr_x, curr_y = xc, yc
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    for step in path_str.split(","):
        if not step:
            continue
        d_idx, dist = map(int, step.split(":"))
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
            if sign == "P":
                img = np.where(mask, img + step, img)
            else:
                img = np.where(mask, img - step, img)
            return np.clip(img, lo, hi).astype(np.float32, copy=False)
        return np.clip(img, lo, hi).astype(np.float32, copy=False)
    else:
        mask = np.zeros((h, w), dtype=bool)
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            num = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0)), 1) + 1
            xs = np.round(np.linspace(x0, x1, num)).astype(int)
            ys = np.round(np.linspace(y0, y1, num)).astype(int)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            mask[ys[valid], xs[valid]] = True
        if sign == "P":
            img = np.where(mask, img + step, img)
        else:
            img = np.where(mask, img - step, img)
        return np.clip(img, lo, hi).astype(np.float32, copy=False)

    return img


def apply_phenotype_to_image_rgb(
    phenotype: str,
    image_chw: np.ndarray,
    clip: tuple[float, float] | None = None,
    turtle_delta_scale: float = 5.0,
) -> np.ndarray:
    """
    ``image_chw``: (3, H, W) Alibi. Phenotype patches: ``Shape-R-...``, ``Shape-G-...``, ``Shape-B-...``.
    """
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"image_chw must be (3,H,W), got {image_chw.shape}")

    _SHAPE_RE = re.compile(r"^Shape-(R|G|B)-")
    _TURTLE_RE = re.compile(
        r"X-(\d+)-Y-(\d+)-Path-([\d:,]+)-Fill-(True|False)-Change-([PN])-(\d+\.\d+)"
    )
    out = image_chw.copy()
    patches = [p for p in phenotype.split("__") if p]
    for patch in patches:
        mshape = _SHAPE_RE.match(patch)
        if not mshape:
            return out
        ch = mshape.group(1)
        cidx = _CHANNEL_TO_IDX[ch]
        patch_body = patch[mshape.end() :]
        plane = out[cidx].copy()
        for m in _TURTLE_RE.finditer(patch_body):
            fill = m.group(4) == "True"
            plane = apply_turtle_to_image_2d(
                plane,
                int(m.group(1)),
                int(m.group(2)),
                m.group(3),
                m.group(5),
                float(m.group(6)),
                fill=fill,
                delta_scale=turtle_delta_scale,
            )
        out[cidx] = plane

    if clip is not None:
        lo, hi = clip
        out = np.clip(out, lo, hi)
    return out


def process_phenotypes_for_images_rgb(
    individuals,
    image_chw: np.ndarray,
    clip: tuple[float, float] | None = None,
    turtle_delta_scale: float = 5.0,
) -> np.ndarray:
    phenos = []
    for ind in individuals:
        phenos.append(ind if isinstance(ind, str) else ind.phenotype)
    imgs = [
        apply_phenotype_to_image_rgb(p, image_chw, clip=clip, turtle_delta_scale=turtle_delta_scale)
        for p in phenos
    ]
    return np.stack(imgs, axis=0)


def alibi_chw_to_hwc_01(img_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) Alibi -> (H,W,3) in [0,1] for matplotlib RGB."""
    x = np.clip(np.transpose(img_chw, (1, 2, 0)) + 0.5, 0.0, 1.0)
    return x.astype(np.float32, copy=False)
