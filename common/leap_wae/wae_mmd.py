"""
WAE_MMD from LeapFactual (Apache-2.0).
https://github.com/caicairay/LeapFactual/blob/main/training_scripts/models/VAE/wae_mmd.py
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .losses import GANLoss, NLayerDiscriminator, VGGPerceptualLoss_Galaxy


def get_norm_layer(channels: int, norm_type: str = "bn") -> nn.Module:
    if norm_type == "bn":
        return nn.BatchNorm2d(channels, eps=1e-4)
    if norm_type == "gn":
        return nn.GroupNorm(8, channels, eps=1e-4)
    raise ValueError("norm_type must be bn or gn")


class ResDown(nn.Module):
    def __init__(self, channel_in: int, channel_out: int, kernel_size: int = 3, norm_type: str = "bn") -> None:
        super().__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        self.conv1 = nn.Conv2d(channel_in, max(channel_out // 2, 1) + channel_out, kernel_size, 2, kernel_size // 2)
        self.norm2 = get_norm_layer(max(channel_out // 2, 1), norm_type=norm_type)
        self.conv2 = nn.Conv2d(max(channel_out // 2, 1), channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.LeakyReLU()
        self.channel_out = channel_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fnc(self.norm1(x))
        x_cat = self.conv1(x)
        skip = x_cat[:, : self.channel_out]
        x = x_cat[:, self.channel_out :]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip


class ResUp(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        kernel_size: int = 3,
        scale_factor: int = 2,
        norm_type: str = "bn",
    ) -> None:
        super().__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        self.conv1 = nn.Conv2d(channel_in, max(channel_in // 2, 1) + channel_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(max(channel_in // 2, 1), norm_type=norm_type)
        self.conv2 = nn.Conv2d(max(channel_in // 2, 1), channel_out, kernel_size, 1, kernel_size // 2)
        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.LeakyReLU()
        self.channel_out = channel_out

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))
        x_cat = self.conv1(x)
        skip = x_cat[:, : self.channel_out]
        x = x_cat[:, self.channel_out :]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip


class ResBlock(nn.Module):
    def __init__(self, channel_in: int, channel_out: int, kernel_size: int = 3, norm_type: str = "bn") -> None:
        super().__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        first_out = max(channel_in // 2, 1) if channel_in == channel_out else max(channel_in // 2, 1) + channel_out
        self.conv1 = nn.Conv2d(channel_in, first_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(max(channel_in // 2, 1), norm_type=norm_type)
        self.conv2 = nn.Conv2d(max(channel_in // 2, 1), channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.LeakyReLU()
        self.skip = channel_in == channel_out
        self.bttl_nk = max(channel_in // 2, 1)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.act_fnc(self.norm1(x_in))
        x_cat = self.conv1(x)
        x = x_cat[:, : self.bttl_nk]
        skip = x_in if self.skip else x_cat[:, self.bttl_nk :]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip


class WAE_MMD(nn.Module):
    def __init__(
        self,
        in_channels: int,
        total_latent_dim: int,
        num_classes: int,
        img_size: int,
        hidden_dims: list[int] | None = None,
        reg_weight: float = 100.0,
        kernel_type: str = "imq",
        latent_var: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 32]
        self.num_classes = num_classes
        self.total_latent_dim = total_latent_dim
        self.img_size = img_size
        self.num_layers = len(hidden_dims)
        spatial = img_size // (2**self.num_layers)
        self.embedding_shape = [hidden_dims[-1], spatial, spatial]
        self.embedding_length = int(torch.prod(torch.tensor(self.embedding_shape)).item())
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var
        self.perceptual_loss = VGGPerceptualLoss_Galaxy()
        self.recon_discriminator = NLayerDiscriminator(input_nc=in_channels, n_layers=3)
        self.gan_loss = GANLoss(gan_mode="lsgan")

        local_in_channels = in_channels
        modules: list[nn.Module] = []
        for h_dim in hidden_dims:
            modules.append(ResBlock(local_in_channels, local_in_channels))
            modules.append(ResDown(local_in_channels, h_dim))
            local_in_channels = h_dim
        modules.append(ResBlock(local_in_channels, local_in_channels))
        self.encoder = nn.Sequential(*modules)

        self.fc_in = nn.Sequential(nn.Linear(self.embedding_length, 512), nn.Linear(512, total_latent_dim))
        self.fc_out = nn.Sequential(nn.Linear(total_latent_dim, 512), nn.Linear(512, self.embedding_length))

        dec_dims = list(reversed(hidden_dims))
        if self.num_classes > 0:
            self.fc_cond = nn.Sequential(
                nn.Linear(self.num_classes, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, self.embedding_length),
            )
            self.cond_embed = nn.ModuleList(
                [
                    nn.Sequential(
                        ResBlock(dec_dims[i], dec_dims[i]),
                        ResUp(
                            dec_dims[i],
                            channel_out=dec_dims[i + 1] if (i + 1) < len(dec_dims) else 8,
                        ),
                    )
                    for i in range(len(dec_dims))
                ]
            )
            self.z_discriminator = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(self.total_latent_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, num_classes),
            )

        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    ResBlock(dec_dims[i] * (2 if num_classes > 0 else 1), dec_dims[i]),
                    ResUp(dec_dims[i], channel_out=dec_dims[i + 1] if (i + 1) < len(dec_dims) else 8),
                )
                for i in range(len(dec_dims))
            ]
        )
        self.output = nn.Sequential(
            ResBlock(8, 8),
            nn.Conv2d(in_channels=8, out_channels=in_channels, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
        )

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        result = torch.flatten(self.encoder(input), start_dim=1)
        return self.fc_in(result)

    def decode(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if cond is not None:
            cond_middle = self.fc_cond(cond).view(-1, *self.embedding_shape)
            decoder_middle = self.fc_out(z).view(-1, *self.embedding_shape)
            for l_cond, l_dec in zip(self.cond_embed, self.decoder):
                decoder_input = torch.cat([decoder_middle, cond_middle], dim=1)
                decoder_middle = l_dec(decoder_input)
                cond_middle = l_cond(cond_middle)
            return self.output(decoder_middle)
        result = self.fc_out(z).view(-1, *self.embedding_shape)
        for layer in self.decoder:
            result = layer(result)
        return self.output(result)

    def forward(self, input: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        labels = kwargs.get("labels")
        if self.num_classes > 0 and labels is not None:
            labels = F.one_hot(labels.squeeze(), self.num_classes).to(torch.float)
        else:
            labels = None
        z = self.encode(input)
        recon_img = self.decode(z, labels)
        return [recon_img, z]

    def forward_gan(self, input: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        labels = kwargs.get("labels")
        if self.num_classes > 0 and labels is not None:
            labels = F.one_hot(labels.squeeze(), self.num_classes).to(torch.float)
        else:
            labels = None
        with torch.no_grad():
            z = self.encode(input)
            recon_img = self.decode(z, labels)
        if labels is not None:
            return self.recon_discriminator(recon_img.detach()), self.z_discriminator(z.detach())
        return self.recon_discriminator(recon_img.detach()), None

    def loss_function(self, *args, **kwargs):
        real_img = kwargs["real_img"]
        labels = kwargs.get("labels")
        y_true = (
            F.one_hot(labels.squeeze(), self.num_classes).to(torch.float)
            if self.num_classes > 0 and labels is not None
            else None
        )
        perceptual_weight = kwargs["M_P"]
        recon_weight = kwargs["M_R"]
        gan_weight = kwargs["M_G"]
        adv_weight = kwargs["M_A"]

        batch_size = real_img.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.reg_weight / max(bias_corr, 1)

        recon_img = args[0]
        z = args[1]

        losses: dict[str, torch.Tensor] = {}
        if recon_weight > 0:
            losses["Reconstruction_Loss"] = recon_weight * F.mse_loss(recon_img, real_img)
        if perceptual_weight > 0:
            losses["Perceptual_Loss"] = perceptual_weight * self.perceptual_loss(recon_img, real_img)
        if reg_weight > 0:
            losses["MMD_Loss"] = self.compute_mmd(z, reg_weight)
        if gan_weight > 0:
            losses["GAN_Loss"] = gan_weight * self.gan_loss(
                self.recon_discriminator(recon_img), target_is_real=True
            )
        if adv_weight > 0 and y_true is not None:
            rand_y = torch.randint(0, self.num_classes, (z.shape[0],), device=z.device)
            losses["ADV_Loss"] = adv_weight * F.cross_entropy(
                self.z_discriminator(z),
                F.one_hot(rand_y, self.num_classes).to(dtype=torch.float, device=z.device),
            )

        loss = torch.tensor(0.0, device=real_img.device)
        for v in losses.values():
            loss = loss + v
        losses["loss"] = loss
        return losses

    def loss_adv(self, *args, **kwargs):
        recon_pred = args[0]
        z_pred = args[1]
        labels = kwargs.get("labels")
        y_true = (
            F.one_hot(labels.squeeze(), self.num_classes).to(torch.float)
            if self.num_classes > 0 and labels is not None
            else None
        )
        real_img = kwargs["real_img"]
        recon_fake_loss = self.gan_loss(recon_pred, target_is_real=False)
        recon_real_loss = self.gan_loss(self.recon_discriminator(real_img), target_is_real=True)
        recon_adv_loss = (recon_real_loss + recon_fake_loss) / 2
        if z_pred is not None and y_true is not None:
            z_adv_loss = F.cross_entropy(z_pred, y_true)
            return {"D_loss": z_adv_loss + recon_adv_loss, "Z_D_loss": z_adv_loss, "RECON_D_loss": recon_adv_loss}
        return {"D_loss": recon_adv_loss}

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x, **kwargs)[0]

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        d = x1.size(1)
        n = x1.size(0)
        x1 = x1.unsqueeze(-2).expand(n, n, d)
        x2 = x2.unsqueeze(-3).expand(n, n, d)
        if self.kernel_type == "rbf":
            return self.compute_rbf(x1, x2)
        if self.kernel_type == "imq":
            return self.compute_inv_mult_quad(x1, x2)
        raise ValueError("Undefined kernel type.")

    def compute_rbf(self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        z_dim = x2.size(-1)
        sigma = 2.0 * z_dim * self.z_var
        return torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))

    def compute_inv_mult_quad(self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        z_dim = x2.size(-1)
        c = 2 * z_dim * self.z_var
        kernel = c / (eps + c + (x1 - x2).pow(2).sum(dim=-1))
        return kernel.sum() - kernel.diag().sum()

    def compute_mmd(self, z: torch.Tensor, reg_weight: float) -> torch.Tensor:
        prior_z = torch.randn_like(z)
        return (
            reg_weight * self.compute_kernel(prior_z, prior_z).mean()
            + reg_weight * self.compute_kernel(z, z).mean()
            - 2 * reg_weight * self.compute_kernel(prior_z, z).mean()
        )
