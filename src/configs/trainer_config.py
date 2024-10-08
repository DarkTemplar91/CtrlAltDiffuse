from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

DatasetType = Literal["celebs", "flowers"]
OptimizerType = Literal["adam", "sgd"]


@dataclass
class TrainerConfig:
    dataset_type: DatasetType = "celebs"
    """Type of dataset to be used (e.g., "celebs", "flowers")."""
    dataset_path: Path = Path("./datasets")
    """Path to the dataset directory."""
    checkpoints: Optional[Path] = None
    """Path to load checkpoint of trained model; None if not used."""
    output: Path = Path("./outputs")
    """Path to store the checkpoint of the trained model;"""
    image_dimensions: tuple[int, int] = (256, 256)
    """Input image dimensions (height, width)."""
    batch_size: int = 32
    """Number of samples per batch."""
    epochs: int = 10
    """Number of training epochs."""
    learning_rate: float = 0.001
    """Learning rate for the optimizer."""
    optimizer: OptimizerType = "adam"
    """Optimizer type (e.g., "adam", "sgd")."""
