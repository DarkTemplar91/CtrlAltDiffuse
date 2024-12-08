import torch
import torch.nn as nn

from .residual_block import ResidualBlock


class UpBlock(nn.Module):
    """Up Block used in the U-NET"""
    def __init__(self, width: int, block_depth: int, input_width: int):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # The first residual block has the size of input_width + width because of the skip connections
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(input_width + width, width) if i == 0 else ResidualBlock(width * 2, width)
             for i in range(block_depth)])

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        x = self.up(x)

        for block in self.res_blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x
