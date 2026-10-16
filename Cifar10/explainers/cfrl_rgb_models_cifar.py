"""RGB encoder/decoder for Alibi CounterfactualRL at 32×32."""

from __future__ import annotations

import torch
import torch.nn as nn


class CFRGBEncoderCifar(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(384, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)


class CFRGBDecoderCifar(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 384 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 3, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 384, 4, 4)
        return self.dec(h)
