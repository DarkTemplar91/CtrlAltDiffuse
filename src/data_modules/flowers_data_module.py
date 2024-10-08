import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102

from configs import TrainerConfig


class FlowersDataModule(pl.LightningDataModule):
    def __init__(self, config: TrainerConfig):
        super().__init__()
        self.flowers_train = None
        self.flowers_valid = None
        self.flowers_test = None

        self.data_dir = config.dataset_path
        self.batch_size = config.batch_size
        self.transform = transforms.Compose([
            transforms.Resize(max(config.image_dimensions)),
            transforms.RandomCrop(config.image_dimensions),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def prepare_data(self):
        """Downloads Flowers102 dataset"""
        Flowers102(self.data_dir, split="train", download=True)
        Flowers102(self.data_dir, split="val", download=True)
        Flowers102(self.data_dir, split="test", download=True)

    def setup(self, stage=None):
        train_dataset = Flowers102(self.data_dir, split="train", transform=self.transform)
        val_dataset = Flowers102(self.data_dir, split="val", transform=self.transform)
        test_dataset = Flowers102(self.data_dir, split="test", transform=self.transform)

        # Redistribute dataset
        full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset])
        self.flowers_train, self.flowers_valid, self.flowers_test = torch.utils.data.random_split(
            full_dataset, [0.8, 0.1, 0.1]
        )

    def train_dataloader(self):
        return DataLoader(self.flowers_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.flowers_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.flowers_test, batch_size=self.batch_size)
