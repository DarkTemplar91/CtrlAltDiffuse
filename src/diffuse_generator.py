import tyro

from configs import GeneratorConfig

def main(config: GeneratorConfig):
    print("This script will generate an image")

def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(GeneratorConfig))

if __name__ == '__main__':
    entrypoint()
