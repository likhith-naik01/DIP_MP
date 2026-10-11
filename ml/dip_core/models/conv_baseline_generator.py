"""Plain Encoder-Decoder DIP Baseline Generator without skip connections."""

import torch
import torch.nn as nn
from dip_core.models.blocks import ConvBnAct, DownBlock, UpBlock


class ConvBaselineGenerator(nn.Module):
    """
    Plain Encoder-Decoder DIP Baseline (no skip connections).
    Used as an architectural ablation baseline to prove that skip connections in the U-Net
    improve restoration fidelity and feature preservation.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 1,
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBnAct(in_channels, 16, kernel_size=3, padding=1),
            ConvBnAct(16, 16, kernel_size=3, padding=1),
        )
        self.down1 = DownBlock(16, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 128)
        self.bottleneck = ConvBnAct(128, 128, kernel_size=3, padding=1)

        # Decoder (skip_channels=0)
        self.up4 = UpBlock(128, skip_channels=0, out_channels=128, upsample_mode=upsample_mode)
        self.up3 = UpBlock(128, skip_channels=0, out_channels=64, upsample_mode=upsample_mode)
        self.up2 = UpBlock(64, skip_channels=0, out_channels=32, upsample_mode=upsample_mode)
        self.up1 = UpBlock(32, skip_channels=0, out_channels=16, upsample_mode=upsample_mode)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(
                f"Input spatial dimensions ({h}x{w}) must be divisible by 32. "
                "Please pad or resize the input image prior to forward pass."
            )

        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        b = self.down4(e4)
        b = self.bottleneck(b)

        d4 = self.up4(b, None)
        d3 = self.up3(d4, None)
        d2 = self.up2(d3, None)
        d1 = self.up1(d2, None)

        out = self.final_conv(d1)
        return self.sigmoid(out)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backward-compatible alias
ConvDIPBaseline = ConvBaselineGenerator
