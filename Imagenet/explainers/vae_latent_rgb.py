"""VAE with frozen ResNet-18 trunk → latent Gaussian; decoder upsamples to RGB Alibi range."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import FrozenResNet18Features


class VAELatentRGB(nn.Module):
    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.backbone = FrozenResNet18Features()
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, 512 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(-1, 512, 7, 7)
        y = self.dec(h)
        return F.interpolate(y, size=(299, 299), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
