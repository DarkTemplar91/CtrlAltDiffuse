import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102

from configs import TrainerConfig


class FlowersDataModule(pl.LightningDataModule):
    def __init__(self, config: TrainerConfig):
        super().__init__()
        self.data_dir = config.dataset_path
        self.batch_size = config.batch_size
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_resolution, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

        self.flowers_train = None

    def prepare_data(self):
        """Downloads Flowers102 dataset"""
        Flowers102(self.data_dir, split="train", download=True)
        Flowers102(self.data_dir, split="val", download=True)
        Flowers102(self.data_dir, split="test", download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset = Flowers102(self.data_dir, split="train", transform=self.transform)
            val_dataset = Flowers102(self.data_dir, split="val", transform=self.transform)
            test_dataset = Flowers102(self.data_dir, split="test", transform=self.transform)
            self.flowers_train = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset])

    def train_dataloader(self):
        return DataLoader(self.flowers_train, batch_size=self.batch_size, shuffle=True)
