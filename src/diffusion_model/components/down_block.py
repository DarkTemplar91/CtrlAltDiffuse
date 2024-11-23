import torch
import torch.nn as nn

from src.diffusion_model.components.residual_block import ResidualBlock


class DownBlock(nn.Module):
    def __init__(self, width: int, block_depth: int, input_width: int):
        super(DownBlock, self).__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(input_width if i == 0 else width, width) for i in range(block_depth)])
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)
