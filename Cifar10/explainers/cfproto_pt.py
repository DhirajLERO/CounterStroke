"""Prototypical counterfactuals for CIFAR-10 RGB (continuous images)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KDTree

from ._utils import model_predict_proba
from .cfproto_ae import CifarProtoAE, train_proto_ae_nchw


def _x_to_nchw(X: np.ndarray) -> np.ndarray:
    """(N,H,W,3) -> (N,3,H,W); leave (N,C,H,W) unchanged."""
    if X.ndim == 4 and X.shape[-1] == 3:
        return np.transpose(X.astype(np.float32), (0, 3, 1, 2))
    return X.astype(np.float32)


@dataclass
class CFProtoExplanation:
    adv: np.ndarray
    orig_class: int
    adv_class: int
    target_class: int
    success: bool


class CounterfactualProtoPT:
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        kappa: float = 0.0,
        beta: float = 0.1,
        gamma: float = 100.0,
        theta: float = 100.0,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        c_init: float = 1.0,
        c_steps: int = 2,
        feature_range: Tuple[float, float] = (-0.5, 0.5),
        ae_train_epochs: int = 50,
        ae_batch_size: int = 128,
        num_classes: int = 10,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.lr = learning_rate
        self.max_iterations = max_iterations
        self.c_init = c_init
        self.c_steps = max(1, c_steps)
        self.feature_range = feature_range
        self.ae_train_epochs = ae_train_epochs
        self.ae_batch_size = ae_batch_size
        self.num_classes = num_classes
        self._latent: Dict[int, np.ndarray] = {}
        self._trees: Dict[int, KDTree] = {}
        self._ae: Optional[CifarProtoAE] = None
        self.model.eval()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CounterfactualProtoPT":
        X_nchw = _x_to_nchw(X)
        y_flat = y.reshape(-1)
        if len(X_nchw) != len(y_flat):
            raise ValueError("X and y must contain the same number of samples.")
        if len(X_nchw) == 0:
            raise ValueError("Cannot fit CFProto on an empty dataset.")

        xt = torch.tensor(X_nchw, dtype=torch.float32)
        self._ae = train_proto_ae_nchw(
            xt,
            self.device,
            epochs=self.ae_train_epochs,
            batch_size=self.ae_batch_size,
        )
        self._freeze_ae()

        latent_batches = []
        batch_size = max(1, self.ae_batch_size)
        with torch.no_grad():
            for start in range(0, len(xt), batch_size):
                xb = xt[start : start + batch_size].to(self.device)
                latent_batches.append(self._ae.encode_compact(xb).cpu().numpy())
        latent_all = np.concatenate(latent_batches, axis=0).astype(np.float32, copy=False)
        self._latent = {
            int(c): np.ascontiguousarray(latent_all[y_flat == c], dtype=np.float32)
            for c in np.unique(y_flat)
        }
        self._build_trees()
        return self

    def _freeze_ae(self) -> None:
        if self._ae is None:
            return
        self._ae.eval()
        for parameter in self._ae.parameters():
            parameter.requires_grad_(False)

    def _build_trees(self) -> None:
        self._trees = {
            class_id: KDTree(latents)
            for class_id, latents in self._latent.items()
            if len(latents) > 0
        }

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.feature_range[0], self.feature_range[1])

    def _ae_loss(self, adv_c: torch.Tensor) -> torch.Tensor:
        if self._ae is None or self.gamma <= 0:
            return adv_c.new_zeros(())
        rec = self._ae(adv_c)
        return torch.nn.functional.mse_loss(rec, adv_c)

    def explain(self, x: np.ndarray, target_class: int) -> CFProtoExplanation:
        if target_class not in self._trees:
            raise ValueError(f"No latent prototypes for class {target_class}; call fit() with labeled data.")
        if self._ae is None:
            raise RuntimeError("Call fit() before explain().")

        self.model.eval()
        x_orig = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            factual_latent = self._ae.encode_compact(x_orig).cpu().numpy()
        nearest_index = int(self._trees[target_class].query(factual_latent, k=1)[1][0, 0])
        target_latent = torch.tensor(
            self._latent[target_class][nearest_index],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        nc = self.num_classes
        y_tgt = torch.zeros(1, nc, device=self.device)
        y_tgt[0, target_class] = 1.0

        with torch.no_grad():
            orig_class = int(model_predict_proba(self.model, x_orig)[0].argmax().item())

        adv = x_orig.clone().detach().requires_grad_(True)
        best_success: Optional[torch.Tensor] = None
        best_success_distance = float("inf")
        best_fallback: Optional[torch.Tensor] = None
        best_fallback_margin = -float("inf")
        best_fallback_distance = float("inf")

        inner = max(1, self.max_iterations)

        for s in range(self.c_steps):
            if self.c_steps <= 1:
                c = float(self.c_init)
            else:
                c = float(self.c_init) * (10.0 ** (s / (self.c_steps - 1)))

            for _ in range(inner):
                self.model.zero_grad(set_to_none=True)
                if adv.grad is not None:
                    adv.grad.zero_()

                adv_c = self._clip(adv)
                probs = model_predict_proba(self.model, adv_c)
                target_proba = (probs * y_tgt).sum(dim=-1)
                nontarget_max = torch.max((1.0 - y_tgt) * probs - y_tgt * 1e4, dim=-1).values
                atk = torch.relu(nontarget_max - target_proba + self.kappa)
                delta = adv_c - x_orig
                l1 = torch.abs(delta).sum()
                l2 = (delta**2).sum()
                adv_latent = self._ae.encode_compact(adv_c)
                proto_loss = ((adv_latent - target_latent) ** 2).sum()
                ae_term = self._ae_loss(adv_c)
                loss = c * atk + self.beta * l1 + l2 + self.gamma * ae_term + self.theta * proto_loss
                loss.backward()

                with torch.no_grad():
                    adv -= self.lr * adv.grad
                    adv.copy_(self._clip(adv))
                adv.grad = None

                with torch.no_grad():
                    candidate = self._clip(adv)
                    p = model_predict_proba(self.model, candidate)[0]
                    target_margin = float(
                        (p[target_class] - torch.cat((p[:target_class], p[target_class + 1 :])).max()).item()
                    )
                    candidate_delta = candidate - x_orig
                    distance = float(
                        (self.beta * torch.abs(candidate_delta).sum() + (candidate_delta**2).sum()).item()
                    )
                    if int(p.argmax().item()) == target_class:
                        if distance < best_success_distance:
                            best_success_distance = distance
                            best_success = candidate.detach().clone()
                    elif (
                        target_margin > best_fallback_margin
                        or (
                            target_margin == best_fallback_margin
                            and distance < best_fallback_distance
                        )
                    ):
                        best_fallback_margin = target_margin
                        best_fallback_distance = distance
                        best_fallback = candidate.detach().clone()

        best = best_success if best_success is not None else best_fallback
        if best is None:
            best = self._clip(adv.detach())
        best = self._clip(best)
        with torch.no_grad():
            p = model_predict_proba(self.model, best)[0]
            adv_class = int(p.argmax().item())

        return CFProtoExplanation(
            adv=best.cpu().numpy(),
            orig_class=orig_class,
            adv_class=adv_class,
            target_class=target_class,
            success=adv_class == target_class,
        )

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._ae is None or not self._latent:
            raise RuntimeError("Call fit() before saving a CFProto checkpoint.")
        latent_np = {str(k): np.asarray(v, dtype=np.float32) for k, v in self._latent.items()}
        payload: dict[str, Any] = {
            "version": 3,
            "latent": latent_np,
            "ae": self._ae.state_dict(),
            "init": {
                "kappa": self.kappa,
                "beta": self.beta,
                "gamma": self.gamma,
                "theta": self.theta,
                "learning_rate": self.lr,
                "max_iterations": self.max_iterations,
                "c_init": self.c_init,
                "c_steps": self.c_steps,
                "feature_range": tuple(float(x) for x in self.feature_range),
                "ae_train_epochs": self.ae_train_epochs,
                "ae_batch_size": self.ae_batch_size,
                "num_classes": self.num_classes,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> "CounterfactualProtoPT":
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = int(payload.get("version", 0))
        if version != 3:
            raise ValueError(
                f"Incompatible CFProto checkpoint version {version} in {path}; "
                "expected version 3 with latent KDTree prototypes. Retrain the checkpoint."
            )
        dev = device or torch.device("cpu")
        init = dict(payload["init"])
        saved_range = tuple(float(x) for x in init.get("feature_range", ()))
        if saved_range != (-0.5, 0.5):
            raise ValueError(
                f"Incompatible CFProto feature_range={saved_range} in {path}; "
                "expected (-0.5, 0.5). Retrain the checkpoint."
            )
        obj = cls(model, device=dev, **init)
        obj._latent = {
            int(k): np.ascontiguousarray(v, dtype=np.float32)
            for k, v in payload["latent"].items()
        }
        if payload.get("ae") is None:
            raise ValueError(f"CFProto checkpoint {path} does not contain the required AE state.")
        obj._ae = CifarProtoAE().to(dev)
        obj._ae.load_state_dict(payload["ae"])
        obj._freeze_ae()
        obj._build_trees()
        obj.model.eval()
        return obj
