"""Latent-space CounterfactualProto for RGB 299×299 ImageNet inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree

from ._utils import model_predict_proba
from torch.utils.data import DataLoader

from .cfproto_ae_rgb import DeepConvAE299, train_deep_ae_dataloader, train_deep_ae_nchw


def _x_to_nchw(X: np.ndarray) -> np.ndarray:
    """(N,H,W,3) -> (N,3,H,W); leave (N,C,H,W) unchanged."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4 and X.shape[-1] == 3:
        return np.transpose(X, (0, 3, 1, 2))
    return X


@dataclass
class CFProtoExplanationRGB:
    adv: np.ndarray
    orig_class: int
    adv_class: int
    target_class: int
    success: bool


class CounterfactualProtoRGB:
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
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
        ae_train_epochs: int = 4,
        ae_batch_size: int = 32,
        latent_dim: int = 96,
    ):
        self.model = model
        self.num_classes = int(num_classes)
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
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0 or 384 % self.latent_dim != 0:
            raise ValueError("latent_dim must be a positive divisor of 384 encoder channels.")
        self._latent_by_class: dict[int, np.ndarray] = {}
        self._trees: dict[int, KDTree] = {}
        self._ae: Optional[DeepConvAE299] = None
        self.model.eval()

    def _latent_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self._ae is None:
            raise RuntimeError("Call fit() before computing CFProto latent embeddings.")
        channels = F.adaptive_avg_pool2d(self._ae.enc(x), output_size=1).flatten(1)
        if self.latent_dim == channels.shape[1]:
            return channels
        group_size = channels.shape[1] // self.latent_dim
        return channels.reshape(channels.shape[0], self.latent_dim, group_size).mean(dim=2)

    def _rebuild_trees(self) -> None:
        self._trees = {
            c: KDTree(latents, leaf_size=40)
            for c, latents in self._latent_by_class.items()
            if len(latents) > 0
        }

    def add_target_class_samples(self, target_class: int, x: np.ndarray) -> None:
        """Encode samples for a missing target class and add its latent KDTree index."""
        if self._ae is None:
            raise RuntimeError("Cannot add target samples before fitting the autoencoder.")
        x_nchw = _x_to_nchw(x)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_nchw), self.ae_batch_size):
                xb = torch.as_tensor(
                    x_nchw[start : start + self.ae_batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                chunks.append(self._latent_embedding(xb).cpu().numpy().astype(np.float32))
        if not chunks:
            raise ValueError(f"No samples supplied for target class {target_class}.")
        latents = np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
        self._latent_by_class[int(target_class)] = latents
        self._trees[int(target_class)] = KDTree(latents, leaf_size=40)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CounterfactualProtoRGB":
        X_nchw = _x_to_nchw(X)
        y_flat = y.reshape(-1)
        xt = torch.tensor(X_nchw, dtype=torch.float32)
        self._ae = train_deep_ae_nchw(
            xt,
            self.device,
            epochs=self.ae_train_epochs,
            batch_size=self.ae_batch_size,
        )
        self._ae.requires_grad_(False)
        self._latent_by_class = {}
        for c in np.unique(y_flat):
            self.add_target_class_samples(int(c), X_nchw[y_flat == c])
        self._rebuild_trees()
        return self

    def fit_dataloader(self, loader: DataLoader) -> "CounterfactualProtoRGB":
        """Train the AE, then stream the loader into compact per-class latent arrays."""
        self._ae = train_deep_ae_dataloader(
            loader,
            self.device,
            epochs=self.ae_train_epochs,
        )
        self._ae.requires_grad_(False)
        latent_chunks: dict[int, list[np.ndarray]] = {}
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                zb = self._latent_embedding(xb).cpu().numpy().astype(np.float32)
                labels = yb.detach().cpu().numpy().reshape(-1)
                for c in np.unique(labels):
                    latent_chunks.setdefault(int(c), []).append(zb[labels == c])
        self._latent_by_class = {
            c: np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
            for c, chunks in latent_chunks.items()
        }
        self._rebuild_trees()
        return self

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.feature_range[0], self.feature_range[1])

    def _ae_loss(self, adv_c: torch.Tensor) -> torch.Tensor:
        if self._ae is None or self.gamma <= 0:
            return adv_c.new_zeros(())
        rec = self._ae(adv_c)
        return torch.nn.functional.mse_loss(rec, adv_c)

    def explain(self, x: np.ndarray, target_class: int) -> CFProtoExplanationRGB:
        if target_class not in self._trees:
            raise ValueError(f"No latent index for class {target_class}; fit() first.")
        if self._ae is None:
            raise RuntimeError("Call fit() before explain().")

        self.model.eval()
        x_orig = torch.tensor(_x_to_nchw(x), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            factual_latent = self._latent_embedding(x_orig).cpu().numpy()
        nearest_idx = int(self._trees[target_class].query(factual_latent, k=1)[1][0, 0])
        target_latent = torch.as_tensor(
            self._latent_by_class[target_class][nearest_idx],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        y_tgt = torch.zeros(1, self.num_classes, device=self.device)
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
            c_mult = float(self.c_init) if self.c_steps <= 1 else float(self.c_init) * (
                10.0 ** (s / max(1, self.c_steps - 1))
            )

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
                proto_loss = ((self._latent_embedding(adv_c) - target_latent) ** 2).sum()
                ae_term = self._ae_loss(adv_c)
                loss = c_mult * atk + self.beta * l1 + l2 + self.gamma * ae_term + self.theta * proto_loss
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

        return CFProtoExplanationRGB(
            adv=best.cpu().numpy(),
            orig_class=orig_class,
            adv_class=adv_class,
            target_class=target_class,
            success=adv_class == target_class,
        )

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if self._ae is None:
            raise RuntimeError("Cannot save CFProto before training DeepConvAE299.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        latent_np = {
            str(k): np.ascontiguousarray(v, dtype=np.float32)
            for k, v in self._latent_by_class.items()
        }
        payload: dict[str, Any] = {
            "version": 4,
            "latent_by_class": latent_np,
            "ae": self._ae.state_dict() if self._ae is not None else None,
            "num_classes": self.num_classes,
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
                "latent_dim": self.latent_dim,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        model: nn.Module,
        num_classes: int,
        device: Optional[torch.device] = None,
    ) -> "CounterfactualProtoRGB":
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = int(payload.get("version", 0))
        if version != 4:
            raise ValueError(
                f"Incompatible CFProto RGB checkpoint version {version} in {path}; "
                "expected version 4 with latent KDTree prototypes. Retrain the checkpoint."
            )
        dev = device or torch.device("cpu")
        init = dict(payload["init"])
        saved_range = tuple(float(x) for x in init.get("feature_range", ()))
        if saved_range != (-0.5, 0.5):
            raise ValueError(
                f"Incompatible CFProto RGB feature_range={saved_range} in {path}; "
                "expected (-0.5, 0.5). Retrain the checkpoint."
            )
        nc = int(payload.get("num_classes", num_classes))
        obj = cls(model, num_classes=nc, device=dev, **init)
        if payload.get("ae") is None:
            raise ValueError(f"CFProto latent checkpoint {path} has no DeepConvAE299 state.")
        obj._ae = DeepConvAE299().to(dev)
        obj._ae.load_state_dict(payload["ae"])
        obj._ae.eval()
        obj._ae.requires_grad_(False)
        obj._latent_by_class = {
            int(k): np.ascontiguousarray(v, dtype=np.float32)
            for k, v in payload["latent_by_class"].items()
        }
        obj._rebuild_trees()
        obj.model.eval()
        return obj
