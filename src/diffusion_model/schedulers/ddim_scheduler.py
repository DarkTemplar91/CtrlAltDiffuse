import torch
from tqdm import tqdm


class DDIMScheduler:
    """
    Implementation of the DDIM scheduler for quicker inference.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 max_signal_rate=0.95, min_signal_rate=0.02,
                 device="cuda:0"):
        self.num_timesteps = num_timesteps
        self.max_signal_rate = torch.tensor(max_signal_rate, dtype=torch.float32, device=device)
        self.min_signal_rate = torch.tensor(min_signal_rate, dtype=torch.float32, device=device)

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        self.device = torch.device(device)

    def diffusion_schedule(self, diffusion_times):
        """Creat the noise and signal rates used for the diffusion process"""
        start_angle = torch.acos(self.max_signal_rate)
        end_angle = torch.acos(self.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, model):
        """Remove noise from the image"""
        pred_noises = model(noisy_images, noise_rates ** 2)

        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps, model):
        """Apply iterative implicit reverse diffusion"""
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise

        # Iteratively denoise the image.
        for step in tqdm(range(diffusion_steps)):
            noisy_images = next_noisy_images
            diffusion_times = torch.ones((num_images, 1, 1, 1), device=self.device) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            with torch.no_grad():
                pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, model)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images
