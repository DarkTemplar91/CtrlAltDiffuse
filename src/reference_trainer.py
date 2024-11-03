import torch
from pathlib import Path
from configs import TrainerConfig
from data_modules import CelebsDataModule, FlowersDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl  # PyTorch Lightning

# Import the model class
from reference_diffusion_model import ReferenceDiffusionModel


def get_data_module(config):
    # Choose between CelebA and Flowers102 datasets
    if config.dataset_type == "CelebA":
        return CelebsDataModule(config)
    elif config.dataset_type == "Flowers102":
        return FlowersDataModule(config)
    else:
        raise ValueError(f"Unsupported dataset type: {config.dataset_type}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")  # Vagy "medium" ha a stabilitás a fontosabb
    # Load configuration
    config = TrainerConfig(dataset_type="Flowers102")  # Explicitly setting to use CelebA
    data_module = get_data_module(config)
    data_module.prepare_data()
    data_module.setup()
    # Initialize model as an instance of ReferenceDiffusionModel
    model = ReferenceDiffusionModel()

    # Set up checkpoint directory
    checkpoint_dir = Path(config.output) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set up ModelCheckpoint callback to save top 3 models based on training loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="diffusion_model-psnr2-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="val_psnr",
        mode="max",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_psnr",
        patience=10,
        mode="max",
        verbose=True
    )
    # Initialize the Trainer with specified epochs, precision, and checkpointing
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        precision="32",
        default_root_dir=checkpoint_dir,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # Start training the model with the data module
    trainer.fit(model, datamodule=data_module)
