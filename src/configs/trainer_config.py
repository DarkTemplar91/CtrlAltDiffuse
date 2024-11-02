from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

DatasetType = Literal["CelebA", "Flowers102"]
OptimizerType = Literal["adam", "sgd"]


@dataclass
class TrainerConfig:
    dataset_type: DatasetType = "CelebA"
    """Type of dataset to be used (e.g., "CelebA", "Flowers102")."""
    dataset_path: Path = Path("./datasets")
    """Path to the dataset directory."""
    checkpoints: Optional[Path] = None
    """Path to load checkpoint of trained model; None if not used."""
    output: Path = Path("./outputs")
    """Path to store the checkpoint of the trained model;"""
    image_resolution: int = 256
    """Input image dimensions: the smaller edge of the image"""
    batch_size: int = 32
    """Number of worker subprocesses"""
    num_workers: int = 4
    """Number of samples per batch."""
    epochs: int = 10
    """Number of training epochs."""
    learning_rate: float = 0.001
    """Learning rate for the optimizer."""
    optimizer: OptimizerType = "adam"
    """Optimizer type (e.g., "adam", "sgd")."""
