"""LeapFactual + LeapFactual_R for ImageNet (WAE_MMD latent + 100-way classifier probs)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_GRACO_ROOT = Path(__file__).resolve().parents[2]
if str(_GRACO_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRACO_ROOT))

from common.leap_wae import LeapWaeMmdAdapter, build_leap_wae_adapter

from .cfm_mlp_rgb import MLP_CFM, progressive_leap


@dataclass
class LeapFactualPairExplanationRGB:
    cf: np.ndarray
    cf_robust: np.ndarray
    orig_class: int
    cf_class: int
    cf_robust_class: int
    success: bool
    success_robust: bool


class LeapFactualImageNet:
    def __init__(
        self,
        classifier: nn.Module,
        vae: LeapWaeMmdAdapter,
        cfm: MLP_CFM,
        device: Optional[torch.device] = None,
        latent_dim: int = 64,
        num_classes: int = 100,
        gamma_blend: float = 0.1,
        max_itr_blend: int = 15,
        gamma_robust: float = 0.1,
        max_itr_robust: int = 5,
        n_ode_steps: int = 5,
        t_lower: float = 0.0,
        t_upper: float = 1.0,
        ode_solver: str = "dopri5",
        ode_sensitivity: str = "adjoint",
        ode_atol: float = 1e-4,
        ode_rtol: float = 1e-4,
        mlp_hidden: int = 512,
    ) -> None:
        self.classifier = classifier
        self.vae = vae
        self.cfm = cfm
        self.device = device or torch.device("cpu")
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.gamma_blend = gamma_blend
        self.max_itr_blend = max_itr_blend
        self.gamma_robust = gamma_robust
        self.max_itr_robust = max_itr_robust
        self.n_ode_steps = n_ode_steps
        self.t_lower = t_lower
        self.t_upper = t_upper
        self.ode_solver = ode_solver
        self.ode_sensitivity = ode_sensitivity
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol
        self.mlp_hidden = mlp_hidden
        self.classifier.eval()
        self.vae.eval()
        self.cfm.eval()

    @classmethod
    def from_checkpoints(
        cls,
        classifier: nn.Module,
        vae_path: Path,
        cfm_path: Path,
        device: torch.device,
        latent_dim: int = 64,
        num_classes: int = 100,
        **kwargs,
    ) -> "LeapFactualImageNet":
        vae = build_leap_wae_adapter("imagenet", device=device)
        vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
        hidden = int(kwargs.pop("mlp_hidden", 512))
        n_layers = int(kwargs.pop("mlp_layers", 3))
        cfm = MLP_CFM(
            dim=latent_dim,
            time_varying=True,
            cond=True,
            cond_dim=num_classes,
            w=hidden,
            n_hidden_layers=n_layers,
        ).to(device)
        cfm.load_state_dict(torch.load(cfm_path, map_location=device, weights_only=True))
        return cls(
            classifier,
            vae,
            cfm,
            device=device,
            latent_dim=latent_dim,
            num_classes=num_classes,
            mlp_hidden=hidden,
            **kwargs,
        )

    def explain_nchw(self, x_nchw: np.ndarray, target_class: int) -> LeapFactualPairExplanationRGB:
        self.vae.eval()
        self.cfm.eval()
        self.classifier.eval()
        x = torch.tensor(x_nchw, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p0 = self.classifier.predict_proba(x)[0]
            orig_class = int(p0.argmax().item())

        mu, _logvar = self.vae.encode(x)
        source_points = mu

        target_y = F.one_hot(
            torch.tensor([target_class], device=self.device, dtype=torch.int64),
            self.num_classes,
        ).float()

        def generator(z: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(z)

        def classifier_softmax(imgs: torch.Tensor) -> torch.Tensor:
            return self.classifier.predict_proba(imgs)

        with torch.no_grad():
            traj = progressive_leap(
                self.cfm,
                source_points,
                classifier_softmax,
                generator,
                target_y,
                gamma=self.gamma_blend,
                max_itr=self.max_itr_blend,
                n_steps=self.n_ode_steps,
                t_lower=self.t_lower,
                t_upper=self.t_upper,
                mode="blending",
                solver=self.ode_solver,
                sensitivity=self.ode_sensitivity,
                atol=self.ode_atol,
                rtol=self.ode_rtol,
            )
            z_cf = traj[-1]
            img_cf = generator(z_cf)

            traj_r = progressive_leap(
                self.cfm,
                z_cf,
                classifier_softmax,
                generator,
                target_y,
                gamma=self.gamma_robust,
                max_itr=self.max_itr_robust,
                n_steps=self.n_ode_steps,
                t_lower=self.t_lower,
                t_upper=self.t_upper,
                mode="injection",
                solver=self.ode_solver,
                sensitivity=self.ode_sensitivity,
                atol=self.ode_atol,
                rtol=self.ode_rtol,
            )
            z_r = traj_r[-1]
            img_r = generator(z_r)

        with torch.no_grad():
            cf_cls = int(self.classifier.predict_proba(img_cf)[0].argmax().item())
            cr_cls = int(self.classifier.predict_proba(img_r)[0].argmax().item())

        cf_np = img_cf.detach().cpu().numpy().astype(np.float32)
        cr_np = img_r.detach().cpu().numpy().astype(np.float32)
        return LeapFactualPairExplanationRGB(
            cf=cf_np,
            cf_robust=cr_np,
            orig_class=orig_class,
            cf_class=cf_cls,
            cf_robust_class=cr_cls,
            success=(cf_cls == target_class),
            success_robust=(cr_cls == target_class),
        )
