import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA

from configs import TrainerConfig


class CelebsDataModule(pl.LightningDataModule):
    def __init__(self, config: TrainerConfig):
        super().__init__()
        self.celebs_train = None
        self.celebs_val = None
        self.celebs_test = None

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
        """Downloads CelebA dataset"""
        CelebA(self.data_dir, split="all", download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.celebs_train = CelebA(self.data_dir, split="train", transform=self.transform)
            self.celebs_val = CelebA(self.data_dir, split="valid", transform=self.transform)

        if stage == "test" or stage is None:
            self.celebs_test = CelebA(self.data_dir, split="test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.celebs_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.celebs_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.celebs_test, batch_size=self.batch_size)
