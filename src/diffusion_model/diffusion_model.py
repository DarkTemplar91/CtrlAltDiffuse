import pytorch_lightning as pl
import torch
from torchvision import transforms
from torchmetrics.image import FrechetInceptionDistance, InceptionScore, KernelInceptionDistance


class DiffusionModel(pl.LightningModule):
    def __init__(self, model, real_images, batch_size=32):
        super(DiffusionModel, self).__init__()
        self.model = model
        self.real_images = real_images
        self.batch_size = batch_size

        # Should we have different instances for training/evaluation/test?
        self.fid_metric = FrechetInceptionDistance(feature=64, normalize=True)
        self.is_metric = InceptionScore(normalize=True)
        self.kid_metric = KernelInceptionDistance(subset_size=50, normalize=True)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def calculate_metrics(self, generated_images):
        self.fid_metric.update(generated_images, real=False)
        fid_score = self.fid_metric.compute()

        self.is_metric.update(generated_images)
        is_score = self.is_metric.compute()

        self.kid_metric.update(generated_images, real=False)
        kid_score = self.kid_metric.compute()

        return fid_score, is_score, kid_score

    def sample_images(self, num_samples):
        # Sample images from the diffusion model
        # TODO: Implement model
        return self.model.sample(num_samples)

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
