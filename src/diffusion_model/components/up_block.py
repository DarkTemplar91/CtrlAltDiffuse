import torch
import torch.nn as nn

from src.diffusion_model.components.residual_block import ResidualBlock


class UpBlock(nn.Module):
    def __init__(self, width: int, block_depth: int, input_width: int):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(input_width if i == 0 else width, width) for i in range(block_depth)])

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        x = self.up(x)

        for block in self.res_blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x
