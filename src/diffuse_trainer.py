from pathlib import Path

import torch
import tyro.extras
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

from data_modules import CelebsDataModule, FlowersDataModule
from configs import TrainerConfig
from diffusion_model.components.unet import UNet
from diffusion_model.diffusion_model import DiffusionModel


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
    print(f"This script will train the model using the dataset '{config.dataset_type}'.")

    # Set up checkpoint directory
    checkpoint_dir = Path(config.output) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    if config.checkpoints:
        diffusion_model = DiffusionModel.load_from_checkpoint(config.checkpoints)
    else:
        unet_model = UNet(
            input_channels=3,
            output_channels=3,
            widths=[32, 64, 96, 128],
            block_depth=2,
            embedding_min_frequency=1e-2,
            embedding_max_frequency=1e4,
            embedding_dims=32,
            device=device
        )

        diffusion_model = DiffusionModel(
            model=unet_model,
            optim_type=config.optimizer,
            device=device,
            ema=0.7
        )

    diffusion_model.train()
    diffusion_model.to(device)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="diffusion_model-loss-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="val_psnr",
        mode="max"
    )

    """early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="max",
        verbose=True
    )"""

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        precision="bf16-mixed",
        default_root_dir=checkpoint_dir,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(diffusion_model, datamodule=data_module)


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(TrainerConfig))


if __name__ == '__main__':
    entrypoint()
