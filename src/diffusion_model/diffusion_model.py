import pytorch_lightning as pl
import torch
import torch.nn.functional as F


def _linear_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_timesteps)


class DiffusionModel(pl.LightningModule):
    def __init__(self, model, num_timesteps=1000, batch_size=32, lr=1e-4):
        super(DiffusionModel, self).__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.lr = lr

        # Noise schedule (betas and precomputed coefficients)
        self.betas = _linear_beta_schedule(num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(torch.tensor(self.alphas), dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Metrics
        #self.fid_metric = FrechetInceptionDistance(feature=64, normalize=True)
        #self.is_metric = InceptionScore(normalize=True)
        #self.kid_metric = KernelInceptionDistance(subset_size=50, normalize=True)

    def forward_diffusion(self, x_0, t):
        batch_size = x_0.shape[0]
        t = t.to(self.device)
        noise = torch.randn_like(x_0).to(self.device)

        alpha_t = self.sqrt_alphas_cumprod[t].view(batch_size, 1, 1, 1).to(self.device)
        alpha_t_bar = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1).to(self.device)

        x_t = alpha_t * x_0 + alpha_t_bar * noise
        return x_t, noise

    def sample_images(self, num_samples, resolution):
        shape = (num_samples, 3, resolution, resolution)
        x_t = torch.randn(shape).to(self.device)

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

            pred_noise = self.model(x_t, t_tensor)

            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            x_0_pred = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

            if t > 0:
                alpha_t_prev = self.alphas_cumprod[t - 1]
                beta_t = self.betas[t]
                noise = torch.randn_like(x_t)
                x_t = (
                    torch.sqrt(alpha_t_prev) * x_0_pred +
                    torch.sqrt(1 - alpha_t_prev) * noise
                )
            else:
                x_t = x_0_pred

        return x_t

    def training_step(self, batch, batch_idx):
        x_0 = batch
        t = torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device)
        x_t, noise = self.forward_diffusion(x_0, t)

        pred_noise = self.model(x_t, t)

        loss = F.mse_loss(pred_noise, noise)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_0 = batch
        t = torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device)
        x_t, noise = self.forward_diffusion(x_0, t)

        pred_noise = self.model(x_t, t)
        val_loss = F.mse_loss(pred_noise, noise)

        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        generated_images = self.sample_images(self.batch_size)
        fid_score, is_score, kid_score = self.calculate_metrics(generated_images)
        self.log('fid_score', fid_score)
        self.log('inception_score', is_score)
        self.log('kid_score', kid_score)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def calculate_metrics(self, generated_images):
        return None, None, None
        """self.fid_metric.update(generated_images, real=False)
        fid_score = self.fid_metric.compute()

        self.is_metric.update(generated_images)
        is_score = self.is_metric.compute()

        self.kid_metric.update(generated_images, real=False)
        kid_score = self.kid_metric.compute()

        return fid_score, is_score, kid_score"""
