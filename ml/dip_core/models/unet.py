import torch
import torch.nn as nn
from dip_core.models.blocks import ConvBlock, DownBlock, UpBlock, Skip1x1


class UNetDIP(nn.Module):
    """
    5-level Deep Image Prior U-Net Generator.

    Architecture specifications:
      - Encoder channels: 16 -> 32 -> 64 -> 128 -> 128 (with stride-2 downsampling)
      - Decoder channels mirror the encoder: 128 -> 64 -> 32 -> 16
      - Skip connections go through 1x1 convs reducing each to 4 channels (DIP paper design)
      - Bilinear upsampling in decoder
      - Final 1x1 conv + Sigmoid activation to ensure [0, 1] range
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 1,
        skip_channels: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels

        # --- Encoder Level 1 (H x W) ---
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, 16, kernel_size=3, padding=1),
            ConvBlock(16, 16, kernel_size=3, padding=1),
        )
        self.skip1 = Skip1x1(16, skip_channels)

        # --- Encoder Level 2 (H/2 x W/2) ---
        self.down1 = DownBlock(16, 32)
        self.skip2 = Skip1x1(32, skip_channels)

        # --- Encoder Level 3 (H/4 x W/4) ---
        self.down2 = DownBlock(32, 64)
        self.skip3 = Skip1x1(64, skip_channels)

        # --- Encoder Level 4 (H/8 x W/8) ---
        self.down3 = DownBlock(64, 128)
        self.skip4 = Skip1x1(128, skip_channels)

        # --- Encoder Level 5 / Bottleneck (H/16 x W/16) ---
        self.down4 = DownBlock(128, 128)
        self.bottleneck = ConvBlock(128, 128, kernel_size=3, padding=1)

        # --- Decoder Levels ---
        # Level 4: 128 up + 4 skip -> 128
        self.up4 = UpBlock(128, skip_channels, 128)
        # Level 3: 128 up + 4 skip -> 64
        self.up3 = UpBlock(128, skip_channels, 64)
        # Level 2: 64 up + 4 skip -> 32
        self.up2 = UpBlock(64, skip_channels, 32)
        # Level 1: 32 up + 4 skip -> 16
        self.up1 = UpBlock(32, skip_channels, 16)

        # Final output projection
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Decoder with skip connections
        d4 = self.up4(b, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        out = self.final_conv(d1)
        return self.sigmoid(out)
