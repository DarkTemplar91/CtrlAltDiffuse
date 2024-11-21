import torch
import diffusers
import pytorch_lightning as pl  # PyTorch Lightning
import torchmetrics.image


class ReferenceDiffusionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Define the U-Net model architecture for diffusion
        self.model = diffusers.models.UNet2DModel(
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            sample_size=256,
        )
        # Noise scheduler for diffusion steps
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            variance_type="fixed_large",
            clip_sample=False,
            num_train_timesteps = 1000
        )

        self.psnr_metric = torchmetrics.image.PeakSignalNoiseRatio()
        self.ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        noise = torch.randn_like(images, dtype=torch.float32)
        steps = torch.randint(
            0, self.scheduler.num_train_timesteps, (images.size(0),), device=self.device, dtype=torch.int64
        ).long()

        # Add noise to images based on the diffusion steps
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        predicted_noise = self.model(noisy_images, steps).sample

        # Compute mean squared error (MSE) loss
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        self.log("train_loss", loss, prog_bar=True)

        psnr = self.psnr_metric(predicted_noise, noise)
        self.log("train_psnr", psnr, prog_bar=True)

        ssim = self.ssim_metric(predicted_noise, noise)
        self.log("train_ssim", ssim, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        noise = torch.randn_like(images)
        steps = torch.randint(
            0, self.scheduler.num_train_timesteps, (images.size(0),), device=self.device , dtype=torch.int32
        )

        noisy_images = self.scheduler.add_noise(images, noise, steps)
        predicted_noise = self.model(noisy_images, steps).sample

        val_loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        self.log("val_loss", val_loss, prog_bar=True)

        val_psnr = self.psnr_metric(predicted_noise, noise)
        val_ssim = self.ssim_metric(predicted_noise, noise)

        self.log("val_psnr", val_psnr, prog_bar=True)
        self.log("val_ssim", val_ssim, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        # Set up the optimizer (AdamW) and learning rate scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        return [optimizer], [scheduler]
