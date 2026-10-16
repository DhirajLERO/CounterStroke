"""PyTorch port of Alibi's Wachter-style Counterfactual (gradient-based CF on the input)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ._utils import model_predict_proba


@dataclass
class CounterfactualExplanation:
    cf: np.ndarray
    orig_class: int
    cf_class: int
    distance_l1: float
    success: bool


def _class_score(
    probs: torch.Tensor,
    orig_class: int,
    target_class: Union[Literal["same", "other"], int],
) -> Tuple[torch.Tensor, Optional[int]]:
    """Scalar score matched to Alibi Counterfactual (counterfactual.py loss_pred)"""
    if target_class == "other":
        mask = torch.ones_like(probs)
        mask[0, orig_class] = 0.0
        score = (probs * mask).max(dim=-1).values
        return score.squeeze(), None
    if target_class == "same":
        return probs[0, orig_class], orig_class
    t = int(target_class)
    return probs[0, t], t


def _tol_met(
    probs: torch.Tensor,
    orig_class: int,
    target_class: Union[Literal["same", "other"], int],
    target_proba: float,
    tol: float,
) -> bool:
    score, _ = _class_score(probs, orig_class, target_class)
    return abs(float(score.item()) - target_proba) <= tol


class CounterfactualPT:
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        distance_fn: str = "l1",
        target_proba: float = 1.0,
        target_class: Union[str, int] = "other",
        max_iter: int = 1000,
        early_stop: int = 50,
        lam_init: float = 0.1,
        max_lam_steps: int = 10,
        tol: float = 0.01,
        learning_rate_init: float = 0.1,
        feature_range: Tuple[float, float] = (-1e10, 1e10),
        decay: bool = True,
    ):
        if distance_fn != "l1":
            raise ValueError("Only l1 distance is implemented.")
        self.model = model
        self.device = device or torch.device("cpu")
        self.target_proba = target_proba
        self.target_class = target_class
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.lam_init = lam_init
        self.max_lam_steps = max_lam_steps
        self.tol = tol
        self.lr0 = learning_rate_init
        self.feature_range = feature_range
        self.decay = decay
        self.model.eval()

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.feature_range
        return torch.clamp(x, lo, hi)

    def _run_inner(
        self,
        x_orig: torch.Tensor,
        lam: float,
        x_init: torch.Tensor,
        orig_class: int,
    ) -> Tuple[torch.Tensor, bool]:
        """Adam on input (like Alibi tf.train.AdamOptimizer on cf variable)."""
        x_cf = nn.Parameter(self._clip(x_init.clone().detach()).to(self.device))
        opt = torch.optim.Adam([x_cf], lr=self.lr0)
        target = self.target_class

        best = x_cf.detach().clone()
        best_dist = float("inf")
        found = False

        for i in range(self.max_iter):
            if self.decay:
                lr_scale = (1.0 - i / max(self.max_iter, 1)) ** 1.0
                for pg in opt.param_groups:
                    pg["lr"] = self.lr0 * lr_scale

            opt.zero_grad(set_to_none=True)
            x_clipped = self._clip(x_cf)
            probs = model_predict_proba(self.model, x_clipped)
            score, _ = _class_score(probs, orig_class, target)
            loss_pred = (score - self.target_proba) ** 2
            dist = torch.abs(x_clipped - x_orig).sum()
            loss = loss_pred + lam * dist
            loss.backward()
            opt.step()

            with torch.no_grad():
                x_cf.copy_(self._clip(x_cf))

            with torch.no_grad():
                p = model_predict_proba(self.model, self._clip(x_cf))[0]
                ok = _tol_met(p.unsqueeze(0), orig_class, target, self.target_proba, self.tol)
                d = float(torch.abs(self._clip(x_cf) - x_orig).sum().item())
                if ok and d < best_dist:
                    best = self._clip(x_cf).detach().clone()
                    best_dist = d
                    found = True

        final = self._clip(x_cf.detach())
        if not found:
            best = final
            best_dist = float(torch.abs(final - x_orig).sum().item())
        return best, found

    def explain(self, x: np.ndarray) -> CounterfactualExplanation:
        """x: NCHW batch (e.g. ``(1, 3, 299, 299)`` ImageNet or ``(1, 1, 28, 28)`` MNIST)."""
        self.model.eval()
        x_orig = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p0 = model_predict_proba(self.model, x_orig)[0]
            pred_class = int(p0.argmax().item())

        lams = np.array([self.lam_init / 10**i for i in range(10)])
        x_current = x_orig.clone()
        cf_count = np.zeros_like(lams, dtype=int)

        for ix, lam_v in enumerate(lams):
            x_current, found = self._run_inner(x_orig, float(lam_v), x_current, pred_class)
            if found:
                cf_count[ix] = 1

        try:
            lb_ix = int(np.where(cf_count > 0)[0][1])
        except IndexError:
            lb_ix = 0
        try:
            ub_ix = int(np.where(cf_count == 0)[0][-1])
        except IndexError:
            ub_ix = 0
        lam = float((lams[lb_ix] + lams[ub_ix]) / 2.0)

        x_current = x_orig.clone()
        best_cf = x_current.clone()
        best_dist = float("inf")
        overall_ok = False

        for _ in range(self.max_lam_steps):
            x_current, _ = self._run_inner(x_orig, lam, x_current, pred_class)
            with torch.no_grad():
                p = model_predict_proba(self.model, self._clip(x_current))[0]
                ok = _tol_met(p.unsqueeze(0), pred_class, self.target_class, self.target_proba, self.tol)
                d = float(torch.abs(x_current - x_orig).sum().item())
                if ok and d < best_dist:
                    best_dist = d
                    best_cf = x_current.clone()
                    overall_ok = True

        if not overall_ok:
            best_cf = self._clip(x_current.clone())

        best_cf = self._clip(best_cf)
        with torch.no_grad():
            p_cf = model_predict_proba(self.model, best_cf)[0]
            cf_class = int(p_cf.argmax().item())
            best_dist = float(torch.abs(best_cf - x_orig).sum().item())

        return CounterfactualExplanation(
            cf=best_cf.cpu().numpy(),
            orig_class=pred_class,
            cf_class=cf_class,
            distance_l1=best_dist,
            success=overall_ok,
        )
