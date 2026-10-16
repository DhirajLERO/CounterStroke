"""Latent-space, k-d-tree CounterfactualProto adaptation for MNIST."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KDTree

from ._utils import model_predict_proba
from .cfproto_ae import TinyMNISTAE, train_tiny_ae_nchw


CHECKPOINT_VERSION = 3


def _x_to_nchw(X: np.ndarray) -> np.ndarray:
    """(N,H,W,1) -> (N,1,H,W); leave (N,C,H,W) unchanged."""
    if X.ndim == 4 and X.shape[-1] == 1:
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
        ae_train_epochs: int = 4,
        ae_batch_size: int = 128,
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
        self._latent_by_class: Dict[int, np.ndarray] = {}
        self._trees: Dict[int, KDTree] = {}
        self._ae: Optional[TinyMNISTAE] = None
        self.model.eval()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CounterfactualProtoPT":
        X_nchw = _x_to_nchw(X)
        y_flat = y.reshape(-1)
        if X_nchw.ndim != 4 or X_nchw.shape[1:] != (1, 28, 28):
            raise ValueError(
                f"CFProto fit expects MNIST images shaped (N,28,28,1) or (N,1,28,28); "
                f"got {tuple(X.shape)}."
            )
        if len(X_nchw) == 0:
            raise ValueError("CFProto fit requires at least one training sample.")
        if len(y_flat) != len(X_nchw):
            raise ValueError(
                f"CFProto fit received {len(X_nchw)} images but {len(y_flat)} labels."
            )
        if self.ae_train_epochs < 0:
            raise ValueError("ae_train_epochs must be non-negative.")
        if self.ae_batch_size <= 0:
            raise ValueError("ae_batch_size must be positive.")

        # The encoder is required for both prototype lookup and optimization,
        # even when gamma disables the optional reconstruction regularizer.
        xt = torch.tensor(X_nchw, dtype=torch.float32)
        self._ae = train_tiny_ae_nchw(
            xt,
            self.device,
            epochs=self.ae_train_epochs,
            batch_size=self.ae_batch_size,
        )
        self._freeze_ae()
        latents = self._encode_numpy(X_nchw)
        self._latent_by_class = {}
        for c in np.unique(y_flat):
            class_latents = np.ascontiguousarray(latents[y_flat == c], dtype=np.float32)
            if len(class_latents) == 0:
                continue
            self._latent_by_class[int(c)] = class_latents
        self._rebuild_trees()
        return self

    def _freeze_ae(self) -> None:
        if self._ae is None:
            return
        self._ae.eval()
        for parameter in self._ae.parameters():
            parameter.requires_grad_(False)

    def _encode_numpy(self, X_nchw: np.ndarray) -> np.ndarray:
        if self._ae is None:
            raise RuntimeError("CFProto autoencoder is unavailable; call fit() or load a checkpoint.")
        encoded = []
        with torch.no_grad():
            for start in range(0, len(X_nchw), self.ae_batch_size):
                batch = torch.as_tensor(
                    X_nchw[start : start + self.ae_batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                encoded.append(self._ae.encoder(batch).flatten(start_dim=1).cpu().numpy())
        if not encoded:
            raise ValueError("Cannot encode an empty CFProto training set.")
        return np.ascontiguousarray(np.concatenate(encoded, axis=0), dtype=np.float32)

    def _rebuild_trees(self) -> None:
        if self._ae is None:
            raise RuntimeError("Cannot build latent k-d trees without a TinyMNISTAE.")
        with torch.no_grad():
            expected_dim = int(
                self._ae.encoder(torch.zeros(1, 1, 28, 28, device=self.device)).numel()
            )
        self._trees = {}
        for class_id, latents in self._latent_by_class.items():
            values = np.asarray(latents, dtype=np.float32)
            if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
                raise ValueError(
                    f"Invalid latent array for class {class_id}: expected non-empty 2D data, "
                    f"got shape {values.shape}."
                )
            if values.shape[1] != expected_dim:
                raise ValueError(
                    f"Invalid latent width for class {class_id}: got {values.shape[1]}, "
                    f"expected {expected_dim} from TinyMNISTAE."
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Latent array for class {class_id} contains non-finite values.")
            self._latent_by_class[class_id] = np.ascontiguousarray(values)
            self._trees[class_id] = KDTree(self._latent_by_class[class_id])

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.feature_range[0], self.feature_range[1])

    def _ae_loss(self, adv_c: torch.Tensor) -> torch.Tensor:
        if self._ae is None or self.gamma <= 0:
            return adv_c.new_zeros(())
        rec = self._ae(adv_c)
        return torch.nn.functional.mse_loss(rec, adv_c)

    def explain(self, x: np.ndarray, target_class: int) -> CFProtoExplanation:
        if self._ae is None or not self._trees:
            raise RuntimeError("CFProto is not fitted; call fit() or load a compatible checkpoint.")
        if target_class not in self._trees:
            available = sorted(self._trees)
            raise ValueError(
                f"No latent k-d tree for target class {target_class}; available classes: {available}."
            )
        x_nchw = _x_to_nchw(np.asarray(x))
        if x_nchw.shape != (1, 1, 28, 28):
            raise ValueError(
                f"CFProto explain expects one MNIST image shaped (1,1,28,28) or "
                f"(1,28,28,1); got {tuple(np.asarray(x).shape)}."
            )

        self.model.eval()
        x_orig = torch.tensor(x_nchw, dtype=torch.float32, device=self.device)
        factual_latent = self._encode_numpy(x_nchw)
        _distance, neighbor_index = self._trees[target_class].query(factual_latent, k=1)
        nearest_latent = torch.as_tensor(
            self._latent_by_class[target_class][int(neighbor_index[0, 0])],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        y_tgt = torch.zeros(1, 10, device=self.device)
        if not 0 <= target_class < y_tgt.shape[1]:
            raise ValueError(f"target_class must be in [0, 9], got {target_class}.")
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
                adv_latent = self._ae.encoder(adv_c).flatten(start_dim=1)
                proto_loss = ((adv_latent - nearest_latent) ** 2).sum()
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
        """Save per-class latent arrays and the TinyMNISTAE state."""
        if self._ae is None or not self._latent_by_class:
            raise RuntimeError("Cannot save an unfitted CFProto explainer; call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        latent_np = {
            str(k): np.asarray(v, dtype=np.float32) for k, v in self._latent_by_class.items()
        }
        payload: dict[str, Any] = {
            "version": CHECKPOINT_VERSION,
            "class_latents": latent_np,
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
        """Load from ``save_checkpoint``; ``model`` must be the same classifier used at train time."""
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Malformed CFProto checkpoint {path}: expected a dictionary payload.")
        version = int(payload.get("version", 0))
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"Incompatible CFProto checkpoint version {version} in {path}; "
                f"expected version {CHECKPOINT_VERSION} with latent k-d-tree prototypes. "
                "Retrain the checkpoint."
            )
        dev = device or torch.device("cpu")
        if "init" not in payload or "class_latents" not in payload or payload.get("ae") is None:
            raise ValueError(
                f"Malformed CFProto checkpoint {path}: expected init settings, class_latents, "
                "and an autoencoder state."
            )
        init = dict(payload["init"])
        saved_range = tuple(float(x) for x in init.get("feature_range", ()))
        if saved_range != (-0.5, 0.5):
            raise ValueError(
                f"Incompatible CFProto feature_range={saved_range} in {path}; "
                "expected (-0.5, 0.5). Retrain the checkpoint."
            )
        obj = cls(model, device=dev, **init)
        obj._latent_by_class = {
            int(k): np.asarray(v, dtype=np.float32)
            for k, v in payload["class_latents"].items()
        }
        if not obj._latent_by_class:
            raise ValueError(f"Malformed CFProto checkpoint {path}: class_latents is empty.")
        obj._ae = TinyMNISTAE().to(dev)
        try:
            obj._ae.load_state_dict(payload["ae"])
        except (RuntimeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid TinyMNISTAE state in CFProto checkpoint {path}.") from exc
        obj._freeze_ae()
        obj._rebuild_trees()
        obj.model.eval()
        return obj
