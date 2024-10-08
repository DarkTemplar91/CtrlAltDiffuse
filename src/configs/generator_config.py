from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GeneratorConfig:
    checkpoints: Optional[Path] = None
    """Path to the trained model"""
    image_dimensions: tuple[int, int] = (256, 256)
    """The dimension of the generated image"""
