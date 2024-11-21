from torch import nn
import torch

from src.diffusion_model.components.attention_block import AttentionBlock
from src.diffusion_model.components.residual_block import ResidualBlock


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.conv_up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x


class AttnUpBlock(UpBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.attn = AttentionBlock(out_channels)

    def forward(self, x, skip):
        x = super().forward(x, skip)
        x = self.attn(x)
        return x
