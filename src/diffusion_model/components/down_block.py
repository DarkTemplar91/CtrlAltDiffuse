from torch import nn

from src.diffusion_model.components.attention_block import AttentionBlock
from src.diffusion_model.components.residual_block import ResidualBlock


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.res_block(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class AttnDownBlock(DownBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.attn = AttentionBlock(out_channels)

    def forward(self, x):
        x, skip = super().forward(x)
        x = self.attn(x)
        return x, skip
