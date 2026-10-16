"""Tiny conv AE for MNIST (1×28×28 NCHW), aligned with Alibi cfproto_mnist bottleneck layout."""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyMNISTAE(nn.Module):
    """Encoder → 14×14×1 bottleneck → decoder, MSE reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_tiny_ae_nchw(
    x_nchw: torch.Tensor,
    device: torch.device,
    epochs: int = 4,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> TinyMNISTAE:
    """Train on float tensor (N,1,28,28)."""
    ae = TinyMNISTAE().to(device)
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
            loss = nn.functional.mse_loss(rec, b)
            loss.backward()
            opt.step()
    ae.eval()
    return ae
