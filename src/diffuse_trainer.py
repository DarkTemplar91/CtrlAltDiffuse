import tyro.extras

from data_modules import CelebsDataModule, FlowersDataModule
from configs import TrainerConfig


def main(config: TrainerConfig):
    if config.dataset_type == "CelebA":
        data_module = CelebsDataModule(config)
    elif config.dataset_type == "Flowers102":
        data_module = FlowersDataModule(config)
    else:
        raise ValueError(f"Dataset type '{config.dataset_type}' not supported")

    data_module.prepare_data()
    data_module.setup()
    print(f"This script will train the model using the dataset '{config.dataset_type}'.")


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(TrainerConfig))


if __name__ == '__main__':
    entrypoint()
