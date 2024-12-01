from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GeneratorConfig:
    checkpoints: Path
    """Path to the trained model"""
    image_resolution: int = 256
    """The width/height of the generated image"""
