"""LeapFactual-style conditional flow matching in latent space + Neural ODE transport."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchdyn.core import NeuralODE


class MLP_CFM(nn.Module):
    """Vector field for conditional FM; input is concat(latent, class_probs) + time."""

    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        w: int = 64,
        time_varying: bool = True,
        cond: bool = True,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        if cond and cond_dim is None:
            cond_dim = dim
        in_dim = dim + (1 if time_varying else 0) + (cond_dim if cond else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, w),
            nn.SELU(),
            nn.Linear(w, w),
            nn.SELU(),
            nn.Linear(w, w),
            nn.SELU(),
            nn.Linear(w, out_dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while t.dim() > 1:
            t = t[:, 0]
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1 and t.shape[0] == 1:
            t = t.expand(x.shape[0])
        return self.net(torch.cat([x, t[:, None]], dim=1))


class TorchWrapperODE(nn.Module):
    """Wraps MLP_CFM for torchdyn: dx/dt = step_size * model(t, cat(x, cond))."""

    def __init__(self, model: MLP_CFM, c: torch.Tensor, step_size: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("_c", c)
        self.step_size = step_size

    def forward(self, t: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        c = self._c
        if c.shape[0] == 1 and x.shape[0] > 1:
            c = c.expand(x.shape[0], -1)
        return self.step_size * self.model(t, torch.cat([x, c], dim=-1))


@torch.no_grad()
def leap(
    model: MLP_CFM,
    source_points: torch.Tensor,
    source_y: torch.Tensor,
    target_y: torch.Tensor,
    gamma: float = 1.0,
    n_steps: int = 10,
    t_lower: float = 0.0,
    t_upper: float = 1.0,
    mode: str = "blending",
    solver: str = "dopri5",
    sensitivity: str | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> torch.Tensor:
    """Sensitivity defaults: ``adjoint`` for adaptive solvers (LeapFactual notebook), ``autograd`` for fixed-step."""
    if sensitivity is None:
        sensitivity = "autograd" if solver in ("euler", "rk4", "midpoint") else "adjoint"
    ode_kw: dict = {"sensitivity": sensitivity}
    if solver not in ("euler", "rk4", "midpoint"):
        ode_kw["atol"] = atol
        ode_kw["rtol"] = rtol
    if mode == "blending":
        node_lift = NeuralODE(
            TorchWrapperODE(model, source_y, gamma),
            solver=solver,
            **ode_kw,
        )
        traj_lift = node_lift.trajectory(
            source_points,
            t_span=torch.linspace(t_upper, t_lower, n_steps, device=source_points.device),
        )
        land_start_points = traj_lift[-1].detach().clone()
    else:
        land_start_points = source_points

    node_land = NeuralODE(
        TorchWrapperODE(model, target_y, gamma),
        solver=solver,
        **ode_kw,
    )
    traj_land = node_land.trajectory(
        land_start_points,
        t_span=torch.linspace(t_lower, t_upper, n_steps, device=source_points.device),
    )
    return traj_land


@torch.no_grad()
def progressive_leap(
    model: MLP_CFM,
    source_points: torch.Tensor,
    classifier_softmax: callable,
    generator: callable,
    target_y: torch.Tensor,
    gamma: float = 0.1,
    max_itr: int = 10,
    n_steps: int = 10,
    t_lower: float = 0.0,
    t_upper: float = 1.0,
    mode: str = "blending",
    solver: str = "dopri5",
    sensitivity: str | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> list[torch.Tensor]:
    assert mode in ("blending", "injection")
    traj_list: list[torch.Tensor] = []
    lift_start_points = source_points
    for _itr in range(max_itr):
        imgs = generator(lift_start_points)
        source_y = classifier_softmax(imgs)
        traj_land = leap(
            model,
            lift_start_points,
            source_y,
            target_y,
            gamma=gamma,
            n_steps=n_steps,
            t_lower=t_lower,
            t_upper=t_upper,
            mode=mode,
            solver=solver,
            sensitivity=sensitivity,
            atol=atol,
            rtol=rtol,
        )
        lift_start_points = traj_land[-1]
        traj_list.append(lift_start_points)
    return traj_list
