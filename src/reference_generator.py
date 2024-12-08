import math
import torch
import torchvision as tv
import diffusers
import jsonargparse
from pathlib import Path
from reference_diffusion_model import ReferenceDiffusionModel


def main(
        checkpoint: Path = Path("outputs/checkpoints/diffusion_model-epoch=05-train_loss=0.01.ckpt"),
        num_timesteps: int = 1000,
        num_samples: int = 1,
        seed: int = 0,
):
    """Generates images from a trained reference diffusion model."""

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReferenceDiffusionModel().to(device)

    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scheduler = diffusers.schedulers.DDPMScheduler(variance_type="fixed_large", timestep_spacing="trailing")
    pipe = diffusers.DDPMPipeline(unet=model.model, scheduler=scheduler)
    pipe = pipe.to(device=device)

    with torch.inference_mode():
        (pil_images,) = pipe(
            batch_size=num_samples,
            num_inference_steps=num_timesteps,
            output_type="pil",
            return_dict=False
        )

    # Plotting the generated images in grids
    images = torch.stack([tv.transforms.functional.to_tensor(pil_image) for pil_image in pil_images])
    image_grid = tv.utils.make_grid(images, nrow=math.ceil(math.sqrt(num_samples)))

    filename = "diffusion_model-diffusion_model-psnr2-epoch=03-train_loss=1.13.png"
    tv.utils.save_image(image_grid, filename)
    print(f"Generated images saved to {filename}")


if __name__ == "__main__":
    jsonargparse.CLI(main)
