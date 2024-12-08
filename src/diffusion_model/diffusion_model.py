import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from .components.unet import UNet
from .schedulers.ddim_scheduler import DDIMScheduler
from configs.trainer_config import OptimizerType


class DiffusionModel(pl.LightningModule):
    def __init__(self, model, optim_type: OptimizerType, num_timesteps=1000, batch_size=32, lr=1e-4, ema=0.95,
                 device=torch.device('cuda')):
        super(DiffusionModel, self).__init__()

        self.save_hyperparameters()

        self.model = model
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.lr = lr
        self.scheduler = DDIMScheduler(num_timesteps=num_timesteps, device=device)  # TODO: pass other arguments?
        self.optim_type = optim_type

        # EMA setup
        # For now, we do not use the EMA model for inference, as it yielded bad results
        self.ema_model = UNet(
            input_channels=self.model.input_channels,
            output_channels=self.model.output_channels,
            widths=self.model.widths,
            block_depth=self.model.block_depth,
            embedding_min_frequency=self.model.embedding_min_frequency,
            embedding_max_frequency=self.model.embedding_max_frequency,
            embedding_dims=self.model.embedding_dims,
            device=device
        )
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema = ema

        # Metrics used to evaluate the models performance
        self.psnr_metric = torchmetrics.image.PeakSignalNoiseRatio()
        self.ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)

    def update_ema(self):
        """Updates the ema model weights. Not used as it yields inferior results"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = self.ema * ema_param.data + (1 - self.ema) * param.data

    def forward(self, x, t):
        return self.model(x, t)

    def generate(self, num_images, diffusion_steps, resolution):
        """Generate images using the DDIM scheduler"""
        initial_noise = torch.randn((num_images, 3, resolution, resolution), device=self.device)
        generated_images = self.scheduler.reverse_diffusion(initial_noise, diffusion_steps, self.model)
        return generated_images

    def training_step(self, batch, batch_idx):
        images, _ = batch
        noise = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.scheduler.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noise

        pred_noises, pred_images = self.scheduler.denoise(noisy_images, noise_rates, signal_rates, self.model)

        loss = F.mse_loss(pred_noises, noise)
        psnr_value = self.psnr_metric(pred_images, images)
        ssim_value = self.ssim_metric(pred_images, images)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_psnr", psnr_value, prog_bar=True)
        self.log("train_ssim", ssim_value, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        noise = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.scheduler.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noise

        pred_noises, pred_images = self.scheduler.denoise(noisy_images, noise_rates, signal_rates, self.model)

        loss = F.mse_loss(pred_noises, noise)
        psnr_value = self.psnr_metric(pred_images, images)
        ssim_value = self.ssim_metric(pred_images, images)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_psnr", psnr_value, prog_bar=True)
        self.log("val_ssim", ssim_value, prog_bar=True)

        return loss

    """def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.update_ema()"""

    def configure_optimizers(self):
        match self.optim_type:
            case "SGD":
                return torch.optim.SGD(self.model.parameters(), lr=self.lr)
            case "Adam":
                return torch.optim.Adam(self.model.parameters(), lr=self.lr)
            case "AdamW":
                return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            case _:
                raise ValueError

    """def state_dict(self, **kwargs):
        state = super().state_dict()
        state["ema_model_state"] = self.ema_model.state_dict()
        return state"""

    def load_state_dict(self, state_dict, **kwargs):
        ema_state = state_dict.pop("ema_model_state", None)
        super().load_state_dict(state_dict)
        """if ema_state:
            self.ema_model.load_state_dict(ema_state)
            self.ema_model.to(self.device)"""
