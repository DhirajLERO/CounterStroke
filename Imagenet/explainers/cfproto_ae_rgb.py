"""Deep convolutional AE for RGB 299×299 (Alibi [-0.5, 0.5] NCHW)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DeepConvAE299(nn.Module):
    """Encoder downsamples to 4×4; decoder upsamples + bilinear align to 299×299."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        y = self.dec(z)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y


def train_deep_ae_nchw(
    x_nchw: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> DeepConvAE299:
    """Train on float tensor (N,3,H,W) in Alibi range."""
    ae = DeepConvAE299().to(device)
    ae.train()
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    data = x_nchw.detach().to(device)
    n = len(data)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            b = data[perm[i : i + batch_size]]
            opt.zero_grad(set_to_none=True)
            rec = ae(b)
            loss = F.mse_loss(rec, b)
            loss.backward()
            opt.step()
    ae.eval()
    return ae


def train_deep_ae_dataloader(
    loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
) -> DeepConvAE299:
    ae = DeepConvAE299().to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    for _ in range(epochs):
        ae.train()
        for xb, _yb in loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            rec = ae(xb)
            loss = F.mse_loss(rec, xb)
            loss.backward()
            opt.step()
    ae.eval()
    return ae
