import torch
from pathlib import Path

import tyro

from configs import TrainerConfig
from data_modules import CelebsDataModule, FlowersDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl

# Import the corrected model class
from reference_diffusion_model import ReferenceDiffusionModel


def main(config: TrainerConfig):
    torch.set_float32_matmul_precision("high")
    if config.dataset_type == "CelebA":
        data_module = CelebsDataModule(config)
    elif config.dataset_type == "Flowers102":
        data_module = FlowersDataModule(config)
    else:
        raise ValueError(f"Dataset type '{config.dataset_type}' not supported")

    data_module.prepare_data()
    data_module.setup()
    print(f"This script will train the reference model using the dataset '{config.dataset_type}'.")

    # Initialize model as an instance of ReferenceDiffusionModel
    model = ReferenceDiffusionModel()

    # Set up checkpoint directory
    checkpoint_dir = Path(config.output) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set up ModelCheckpoint callback to save top 3 models based on training loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="diffusion_model-psnr-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        save_last=True,
        monitor="val_psnr",
        mode="max",
    )

    """early_stopping_callback = EarlyStopping(
        monitor="val_psnr",
        patience=15,
        mode="max",
        verbose=True
    )"""

    # Initialize the Trainer with specified epochs, precision, and checkpointing
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        precision="bf16-mixed",
        default_root_dir=checkpoint_dir,
        callbacks=[checkpoint_callback]
    )

    # Start training the model with the data module
    trainer.fit(model, datamodule=data_module)


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(TrainerConfig))


if __name__ == "__main__":
    entrypoint()