import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
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
        self.num_workers = config.num_workers
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config.image_resolution, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize(size=config.image_resolution, antialias=True),
            transforms.CenterCrop(size=config.image_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def prepare_data(self):
        """Downloads CelebA dataset"""
        CelebA(self.data_dir, split="all", download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.celebs_train = CelebA(self.data_dir, split="train", transform=self.transform_train)
            self.celebs_val = CelebA(self.data_dir, split="valid", transform=self.transform_val)

        if stage == "test" or stage is None:
            self.celebs_test = CelebA(self.data_dir, split="test", transform=self.transform_val)

    def train_dataloader(self):
        return DataLoader(self.celebs_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.celebs_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.celebs_test, batch_size=self.batch_size, num_workers=self.num_workers)
