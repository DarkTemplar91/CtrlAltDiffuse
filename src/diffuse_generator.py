import math
from pathlib import Path

import tyro
import torch
import matplotlib.pyplot as plt

from configs import GeneratorConfig
from diffusion_model.diffusion_model import DiffusionModel


def load_model_checkpoint(model, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded model checkpoint from {checkpoint_path}")


def main(config: GeneratorConfig):
    print("Initializing U-Net model...")

    device = torch.device("cuda")

    diffusion_model = DiffusionModel.load_from_checkpoint(config.checkpoints)
    diffusion_model.eval()
    diffusion_model.to(device)

    print(f"Generating a random image at resolution {config.image_resolution}x{config.image_resolution}...")
    num_images = 8
    with torch.inference_mode():
        generated_images = diffusion_model.generate(
            num_images=num_images,
            diffusion_steps=1000,
            resolution=config.image_resolution
        )

    cols = 4
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    axes = axes.flatten()
    generated_images = (generated_images * 0.5) + 0.5
    generated_images = torch.clamp(generated_images, 0, 1)
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
