"""WAE_MMD model + training hyperparameters (galaxy10_vae.yaml baseline)."""

from __future__ import annotations

# Galaxy training recipe (LeapFactual galaxy10_vae.yaml)
GALAXY_TRAIN_PARAMS: dict = {
    "LR": 3e-4,
    "LR_2": 5e-5,
    "weight_decay": 0.001,
    "scheduler_gamma": 0.99,
    "scheduler_gamma_2": 0.99,
    "perceptual_weight": 10.0,
    "gan_weight": 1.0,
    "adv_weight": 0.0,
    "recon_weight": 1.0,
    "gradient_clip_val": 0.1,
    "max_epochs": 300,
    "val_every": 10,
}

# CIFAR-10: 32×32, 5 downsample stages → 1×1 bottleneck
CIFAR10_WAE_MODEL: dict = {
    "in_channels": 3,
    "total_latent_dim": 64,
    "num_classes": 0,
    "img_size": 32,
    "hidden_dims": [32, 64, 128, 256, 512],
    "reg_weight": 10.0,
    "kernel_type": "imq",
    "latent_var": 2.0,
}

# ImageNet WAE trains at 256×256 (6 stages → 4×4 bottleneck); pipeline tensors stay 299×299
IMAGENET_WAE_IMG_SIZE = 256
IMAGENET_PIPELINE_HW = (299, 299)

IMAGENET_WAE_MODEL: dict = {
    "in_channels": 3,
    "total_latent_dim": 64,
    "num_classes": 0,
    "img_size": IMAGENET_WAE_IMG_SIZE,
    "hidden_dims": [32, 64, 128, 256, 512, 512],
    "reg_weight": 10.0,
    "kernel_type": "imq",
    "latent_var": 2.0,
}
