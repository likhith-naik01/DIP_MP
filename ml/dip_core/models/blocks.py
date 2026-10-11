import torch
import torch.nn as nn
from typing import Optional


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + LeakyReLU block with reflection padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="reflect",
                bias=not use_bn,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling block: stride-2 convolution followed by a standard conv block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Skip1x1(nn.Module):
    """1x1 convolution projecting skip connections down to reduced channel count (e.g. 4)."""

    def __init__(self, in_channels: int, out_channels: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class UpBlock(nn.Module):
    """Bilinear upsampling block with skip connection concatenation."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        total_in = in_channels + skip_channels
        self.conv = nn.Sequential(
            ConvBlock(total_in, out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Handle potential 1-pixel rounding difference if input wasn't perfectly padded
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)
