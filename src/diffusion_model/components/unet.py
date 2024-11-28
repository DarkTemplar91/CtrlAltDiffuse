import torch
import torch.nn as nn
import torch.nn.functional as F

from .down_block import DownBlock
from .residual_block import ResidualBlock
from .sinusodial_embeding import SinusoidalEmbedding
from .up_block import UpBlock


class UNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 widths: list,
                 block_depth: int,
                 embedding_min_frequency: float,
                 embedding_max_frequency: float,
                 embedding_dims: int,
                 device=torch.device("cuda:0")):
        super(UNet, self).__init__()
        self.device = device

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.widths = widths
        self.block_depth = block_depth
        self.embedding_min_frequency = embedding_min_frequency
        self.embedding_max_frequency = embedding_max_frequency
        self.embedding_dims = embedding_dims

        self.embedding = SinusoidalEmbedding(
            embedding_min_frequency=embedding_min_frequency,
            embedding_max_frequency=embedding_max_frequency,
            embedding_dims=embedding_dims,
            device=self.device
        )

        #self.embedding_projection = nn.Conv2d(embedding_dims, widths[0], kernel_size=1)

        self.initial_conv = nn.Conv2d(input_channels, widths[0], kernel_size=1)

        # Downsampling layers
        self.down_blocks = nn.ModuleList()
        current_width = widths[0] + embedding_dims
        for width in widths[:-1]:
            self.down_blocks.append(DownBlock(width=width, block_depth=block_depth, input_width=current_width))
            current_width = width

        # Bottleneck
        self.bottleneck = nn.Sequential()
        for idx in range(block_depth):
            input_width = current_width if idx == 0 else widths[-1]
            self.bottleneck.append(ResidualBlock(input_width=input_width, width=widths[-1]))

        # Upsampling layers
        self.up_blocks = nn.ModuleList()
        for idx, width in enumerate(reversed(widths[:-1])):
            input_width = widths[-1] if idx == 0 else current_width
            self.up_blocks.append(UpBlock(width=width, block_depth=block_depth, input_width=input_width))
            current_width = width

        # Final convolution
        self.final_conv = nn.Conv2d(current_width, output_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.final_conv.weight)

    def forward(self, images: torch.Tensor, noise_variances: torch.Tensor) -> torch.Tensor:
        skips = []

        embedding = self.embedding(noise_variances)
        embedding = embedding.permute(0, 3, 1, 2)
        #embedding = self.embedding_projection(embedding)
        embedding = F.interpolate(embedding, size=images.shape[2:], mode="nearest")

        # Initial convolution and concatenate with embedding
        x = self.initial_conv(images)
        x = torch.cat([x, embedding], dim=1)

        # Downsampling path
        for down_block in self.down_blocks:
            x = down_block(x, skips)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling path
        for up_block in self.up_blocks:
            x = up_block(x, skips)

        # Final output layer
        return self.final_conv(x)

