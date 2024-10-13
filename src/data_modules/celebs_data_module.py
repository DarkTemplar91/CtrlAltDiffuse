import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA

from configs import TrainerConfig


class CelebsDataModule(pl.LightningDataModule):
    def __init__(self, config: TrainerConfig):
        super().__init__()
        self.celebs_train = None

        self.data_dir = config.dataset_path
        self.batch_size = config.batch_size
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_resolution, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def prepare_data(self):
        """Downloads CelebA dataset"""
        CelebA(self.data_dir, split="all", download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.celebs_train = CelebA(self.data_dir, split="all", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.celebs_train, batch_size=self.batch_size, shuffle=True)
