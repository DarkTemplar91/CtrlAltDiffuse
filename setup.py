from setuptools import setup, find_packages

setup(
    name="ctrlaltdiffuse",
    version="0.1.0",
    description="Pytorch Lightning implementation of a Deep Diffusion Implicit Model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "diffuse-generate=diffuse_generator:entrypoint",
            "diffuse-train=diffuse_trainer:entrypoint",
        ],
    },
)
