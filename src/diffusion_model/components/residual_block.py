import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_width: int, width: int):
        super(ResidualBlock, self).__init__()
        self.match_channels = (input_width != width)

        if self.match_channels:
            self.residual_conv = nn.Conv2d(input_width, width, kernel_size=1)

        self.bn = nn.BatchNorm2d(input_width, affine=False)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x) if self.match_channels else x

        x = self.bn(x)
        x = F.silu(self.conv1(x))

        x = self.conv2(x)
        return x + residual
