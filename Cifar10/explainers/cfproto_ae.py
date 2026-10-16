"""Conv AE for CFProto on CIFAR RGB (3×32×32 NCHW, Alibi range)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarProtoAE(nn.Module):
    """~CFRL-scale encoder/decoder with a compact 256-D latent view."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode_compact(self, x: torch.Tensor) -> torch.Tensor:
        """Encode NCHW images and deterministically pool spatial features."""
        encoded = self.encoder(x)
        return F.adaptive_avg_pool2d(encoded, output_size=1).flatten(start_dim=1)


# Backward-compatible alias (checkpoints must be retrained after upgrade)
TinyCifarRGBAE = CifarProtoAE


def train_proto_ae_nchw(
    x_nchw: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> CifarProtoAE:
    if epochs < 0:
        raise ValueError("epochs must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    ae = CifarProtoAE().to(device)
    if epochs == 0:
        ae.eval()
        return ae
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


train_tiny_ae_nchw = train_proto_ae_nchw
