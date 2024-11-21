from torch import nn

from src.diffusion_model.components.down_block import DownBlock, AttnDownBlock
from src.diffusion_model.components.residual_block import ResidualBlock
from src.diffusion_model.components.up_block import UpBlock, AttnUpBlock


class AdvancedUNet(nn.Module):
    def __init__(
            self,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            input_channels=3,
            output_channels=3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types

        # Encoder
        self.__create_encoder_block()

        # Bottleneck
        self.bottleneck = ResidualBlock(block_out_channels[-1], block_out_channels[-1] * 2)

        # Decoder
        in_channels = self.__create_decoder_block()

        # Final output layer
        self.final_conv = nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)

        # Output
        x = self.final_conv(x)
        return x

    def __create_encoder_block(self):
        self.down_blocks = nn.ModuleList()
        in_channels = self.input_channels
        for idx, out_channels in enumerate(self.block_out_channels):
            block_type = self.down_block_types[idx]
            if block_type == "DownBlock2D":
                self.down_blocks.append(DownBlock(in_channels, out_channels))
            elif block_type == "AttnDownBlock2D":
                self.down_blocks.append(AttnDownBlock(in_channels, out_channels))
            in_channels = out_channels

    def __create_decoder_block(self):
        self.up_blocks = nn.ModuleList()
        reversed_out_channels = list(reversed(self.block_out_channels))
        in_channels = self.block_out_channels[-1] * 2  # From bottleneck
        for idx, out_channels in enumerate(reversed_out_channels):
            block_type = self.up_block_types[idx]
            if block_type == "UpBlock2D":
                self.up_blocks.append(UpBlock(in_channels, out_channels))
            elif block_type == "AttnUpBlock2D":
                self.up_blocks.append(AttnUpBlock(in_channels, out_channels))
            in_channels = out_channels

        return in_channels
