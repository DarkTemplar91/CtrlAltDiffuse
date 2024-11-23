import torch
import torch.nn as nn

from src.diffusion_model.components.down_block import DownBlock
from src.diffusion_model.components.residual_block import ResidualBlock
from src.diffusion_model.components.up_block import UpBlock


class UNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 base_width: int,
                 num_layers: int,
                 block_depth: int):
        
        super(UNet, self).__init__()
        self.num_layers = num_layers

        # Downsampling layers
        self.down_blocks = nn.ModuleList()
        current_width = input_channels
        for i in range(num_layers):
            next_width = base_width * (2 ** i)
            self.down_blocks.append(DownBlock(width=next_width, block_depth=block_depth, input_width=current_width))
            current_width = next_width

        # Bottleneck
        self.bottleneck = ResidualBlock(input_width=current_width, width=current_width)

        # Upsampling layers
        self.up_blocks = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            next_width = base_width * (2 ** i)
            self.up_blocks.append(UpBlock(width=next_width, block_depth=block_depth, input_width=current_width))
            current_width = next_width

        # Final convolution
        self.final_conv = nn.Conv2d(current_width, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        for down_block in self.down_blocks:
            x = down_block(x, skips)

        x = self.bottleneck(x)

        for up_block in self.up_blocks:
            x = up_block(x, skips)

        x = self.final_conv(x)
        return x
