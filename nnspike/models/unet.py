import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""  # noqa: D415

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: int | None = None
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: bool = True
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle potential size mismatches for 640x480 resolution
        diffY = x2.size()[2] - x1.size()[2]  # noqa: N806
        diffX = x2.size()[3] - x1.size()[3]  # noqa: N806

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Memory-optimized U-Net for 640x480 image.

    Fixed input size: (batch_size, channels, 480, 640)
    Output size: (batch_size, n_classes, 480, 640)

    Memory usage: ~600MB for batch_size=4 during training
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        bilinear: bool = False,
        reduced_channels: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.input_size = (480, 640)  # Height x Width

        # Reduced channel sizes for memory efficiency
        if reduced_channels:
            base_channels = 32  # Reduced from 64
            channels = [
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
                base_channels * 16,
            ]
        else:
            base_channels = 64  # Original size
            channels = [
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
                base_channels * 16,
            ]

        self.inc = DoubleConv(n_channels, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])

        factor = 2 if bilinear else 1
        self.down4 = Down(channels[3], channels[4] // factor)

        self.up1 = Up(channels[4], channels[3] // factor, bilinear)
        self.up2 = Up(channels[3], channels[2] // factor, bilinear)
        self.up3 = Up(channels[2], channels[1] // factor, bilinear)
        self.up4 = Up(channels[1], channels[0], bilinear)

        self.outc = OutConv(channels[0], n_classes)

        # Add input size validation
        self._validate_input_size = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate input size
        if self._validate_input_size:
            expected_size = (x.size(0), self.n_channels, 480, 640)
            if x.size() != expected_size:
                raise ValueError(f"Expected input size {expected_size}, got {x.size()}")

        # Encoder path
        x1 = self.inc(x)  # 640x480 -> 640x480
        x2 = self.down1(x1)  # 640x480 -> 320x240
        x3 = self.down2(x2)  # 320x240 -> 160x120
        x4 = self.down3(x3)  # 160x120 -> 80x60
        x5 = self.down4(x4)  # 80x60 -> 40x30

        # Decoder path
        x = self.up1(x5, x4)  # 40x30 -> 80x60
        x = self.up2(x, x3)  # 80x60 -> 160x120
        x = self.up3(x, x2)  # 160x120 -> 320x240
        x = self.up4(x, x1)  # 320x240 -> 640x480

        logits = self.outc(x)  # 640x480 -> 640x480
        return logits
