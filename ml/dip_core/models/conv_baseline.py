import torch
import torch.nn as nn
from dip_core.models.blocks import ConvBlock, DownBlock, UpBlock


class ConvDIPBaseline(nn.Module):
    """
    Plain Encoder-Decoder DIP Baseline (without skip connections).
    Used as comparative baseline against U-Net DIP.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, 16, kernel_size=3, padding=1),
            ConvBlock(16, 16, kernel_size=3, padding=1),
        )
        self.down1 = DownBlock(16, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 128)
        self.bottleneck = ConvBlock(128, 128, kernel_size=3, padding=1)

        # Decoder (no skip connections: skip_channels=0)
        self.up4 = UpBlock(128, skip_channels=0, out_channels=128)
        self.up3 = UpBlock(128, skip_channels=0, out_channels=64)
        self.up2 = UpBlock(64, skip_channels=0, out_channels=32)
        self.up1 = UpBlock(32, skip_channels=0, out_channels=16)

        # Output projection
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
