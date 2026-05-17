import torch
import torch.nn as nn


class ResidualFeatureTranslator(nn.Module):
    """Residual 3x3-conv translator for dense feature maps."""

    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden_channels = int(hidden_channels or channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.net(x)


class FeatureDiscriminator(nn.Module):
    """Small PatchGAN-style discriminator for encoder feature maps."""

    def __init__(self, channels, base_channels=64):
        super().__init__()
        base_channels = int(base_channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(min(32, base_channels * 2), base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def set_requires_grad(module, requires_grad):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad
