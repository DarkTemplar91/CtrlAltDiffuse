import tyro
import torch
import matplotlib.pyplot as plt

from configs import GeneratorConfig
from src.diffusion_model.components.unet import UNet
from src.diffusion_model.diffusion_model import DiffusionModel


def main(config: GeneratorConfig):
    print("This script will generate an image")

    model = UNet(
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        input_channels=3, output_channels=3, embedding_dim=32
    )

    diffusion_model = DiffusionModel(model, num_timesteps=1000, batch_size=1, lr=1e-4)

    generated_image = diffusion_model.sample_images(1, 128)

    # Plot generated image
    generated_image = generated_image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    plt.imshow((generated_image + 1) / 2)
    plt.axis('off')
    plt.show()


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(GeneratorConfig))


if __name__ == '__main__':
    entrypoint()
