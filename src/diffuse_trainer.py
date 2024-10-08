import tyro.extras

from data_modules import CelebsDataModule, FlowersDataModule
from configs import TrainerConfig


def main(config: TrainerConfig):
    data_module = CelebsDataModule(config) if config.dataset_type == "celebs" else FlowersDataModule(config)
    data_module.prepare_data()
    print(f"This script will train the model using the dataset '{config.dataset_type}'.")


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(TrainerConfig))


if __name__ == '__main__':
    entrypoint()
