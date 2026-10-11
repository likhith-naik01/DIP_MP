"""5-Level Deep Image Prior U-Net Generator with skip connections."""

import torch
import torch.nn as nn
from dip_core.models.blocks import ConvBnAct, DownBlock, UpBlock, SkipProjection


class UNetGenerator(nn.Module):
    """
    5-level Deep Image Prior U-Net Generator.

    Architecture details:
      - Input: random noise tensor z ~ N(0, I) with shape (1, 32, H, W)
      - Encoder: 5 levels with channels (16, 32, 64, 128, 128) using stride-2 downsampling
      - Decoder: Mirrors encoder using bilinear upsampling
      - Skips: 1x1 conv projections (SkipProjection) reducing features to 4 channels
      - Activation: LeakyReLU(0.2), BatchNorm2d, final 1x1 conv + Sigmoid -> [0, 1] output
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 1,
        skip_channels: int = 4,
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.upsample_mode = upsample_mode

        # --- Encoder Level 1 (H x W) ---
        self.enc1 = nn.Sequential(
            ConvBnAct(in_channels, 16, kernel_size=3, padding=1),
            ConvBnAct(16, 16, kernel_size=3, padding=1),
        )
        self.skip1 = SkipProjection(16, skip_channels)

        # --- Encoder Level 2 (H/2 x W/2) ---
        self.down1 = DownBlock(16, 32)
        self.skip2 = SkipProjection(32, skip_channels)

        # --- Encoder Level 3 (H/4 x W/4) ---
        self.down2 = DownBlock(32, 64)
        self.skip3 = SkipProjection(64, skip_channels)

        # --- Encoder Level 4 (H/8 x W/8) ---
        self.down3 = DownBlock(64, 128)
        self.skip4 = SkipProjection(128, skip_channels)

        # --- Encoder Level 5 / Bottleneck (H/16 x W/16) ---
        self.down4 = DownBlock(128, 128)
        self.bottleneck = ConvBnAct(128, 128, kernel_size=3, padding=1)

        # --- Decoder Levels ---
        # Level 4: 128 + 4 skip -> 128
        self.up4 = UpBlock(128, skip_channels, 128, upsample_mode=upsample_mode)
        # Level 3: 128 + 4 skip -> 64
        self.up3 = UpBlock(128, skip_channels, 64, upsample_mode=upsample_mode)
        # Level 2: 64 + 4 skip -> 32
        self.up2 = UpBlock(64, skip_channels, 32, upsample_mode=upsample_mode)
        # Level 1: 32 + 4 skip -> 16
        self.up1 = UpBlock(32, skip_channels, 16, upsample_mode=upsample_mode)

        # Output projection
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(
                f"Input spatial dimensions ({h}x{w}) must be divisible by 32 for 5-level U-Net. "
                "Please pad or resize the input image prior to forward pass."
            )

        # Encoder
        e1 = self.enc1(x)
        s1 = self.skip1(e1)

        e2 = self.down1(e1)
        s2 = self.skip2(e2)

        e3 = self.down2(e2)
        s3 = self.skip3(e3)

        e4 = self.down3(e3)
        s4 = self.skip4(e4)

        b = self.down4(e4)
        b = self.bottleneck(b)

        # Decoder with 1x1 skip connections
        d4 = self.up4(b, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        out = self.final_conv(d1)
        return self.sigmoid(out)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backward-compatible alias
UNetDIP = UNetGenerator
