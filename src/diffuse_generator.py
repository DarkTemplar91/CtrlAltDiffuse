import math
from pathlib import Path

import tyro
import torch
import matplotlib.pyplot as plt

from configs import GeneratorConfig
from src.diffusion_model.components.unet import UNet
from src.diffusion_model.diffusion_model import DiffusionModel


def load_model_checkpoint(model, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded model checkpoint from {checkpoint_path}")


def main(config: GeneratorConfig):
    print("Initializing U-Net model...")

    device = torch.device("cuda")
    unet_model = UNet(
        input_channels=3,
        output_channels=3,
        widths=[32, 64, 96, 128],
        block_depth=2,
        embedding_min_frequency=1e-2,
        embedding_max_frequency=1e4,
        embedding_dims=32,
        device=device
    )

    diffusion_model = DiffusionModel(
        model=unet_model,
        device=device,
    )

    if config.checkpoints:
        load_model_checkpoint(diffusion_model, config.checkpoints)

    diffusion_model.eval()
    diffusion_model.to(device)

    print(f"Generating a random image at resolution {config.image_resolution}x{config.image_resolution}...")
    num_images = 8
    generated_images = diffusion_model.generate(
        num_images=num_images,
        diffusion_steps=1000,
        resolution=config.image_resolution
    )

    cols = 4
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    axes = axes.flatten()
    for i in range(num_images):
        ax = axes[i]
        generated_image = generated_images[i].cpu().detach().numpy().transpose(1, 2, 0)

        ax.imshow(generated_image)
        ax.axis("off")
        ax.set_title(f"Generated Image {i + 1}")

    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.show()

    print("Image generation completed.")


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(GeneratorConfig))


if __name__ == '__main__':
    entrypoint()
