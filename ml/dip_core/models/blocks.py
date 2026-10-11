"""Reusable convolutional and upsampling blocks for DIP generator architectures."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    """Convolution with reflection padding, BatchNorm2d, and LeakyReLU(0.2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling block: stride-2 convolution followed by a standard conv block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            ConvBnAct(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            ConvBnAct(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class SkipProjection(nn.Module):
    """
    1x1 convolution projecting skip connections down to reduced channel count (e.g. 4).
    Following Ulyanov et al., this prevents high-frequency noise from passing trivially through skips.
    """

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
    """Bilinear upsampling block with optional skip-connection concatenation."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        self.upsample_mode = upsample_mode
        total_in = in_channels + skip_channels
        self.conv = nn.Sequential(
            ConvBnAct(total_in, out_channels, kernel_size=3, stride=1, padding=1),
            ConvBnAct(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)
        if skip is not None:
            # Align spatial dimensions in case of any slight rounding mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode=self.upsample_mode, align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)
