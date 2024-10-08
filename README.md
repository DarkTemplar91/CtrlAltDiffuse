# About
A python application for training a basic Deep Diffusion Implicit Model (DDIM) on the CelebsA and Flowers102 dataset.
In addition to training models, this application can also be used to generate custom images based on these datasets.

# Installation
First of all, you must have a CUDA capable video card for this to work. The application was tested with version CUDA 12.1.
## Development
### Python Environment
Create a python environment. Conda is recommended:
```
conda create --name CtrlAltDiffuse
conda activate CtrlAltDiffuse
pip install --upgrade pip
```
Then you must install the correct version of PyTorch. This depends on your installed CUDA version.\
See more information about installing PyTorch in their [Get Started](https://pytorch.org/get-started/locally/) page.\
For CUDA 12.1 run the following command:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
### Clone repository
```
git clone https://github.com/DarkTemplar91/CtrlAltDiffuse.git
cd CtrlAltDiffuse
```
After cloning the repository, you have two options:
- Installing the repository as a python package (Recommended).
  * ```pip install -e .```
- Configuring the python environment.
  * ```pip install -r requirements.txt```

If you opted to install it as a package, you can use the ```diffuse-train``` and ```diffuse-generate```
commands instead of directly calling the corresponding scripts.  


## Docker
### A. Building Docker Image with CUDA 12 and Python 3.12
```bash
docker build -t ctrlaltdiffuse-cuda12 -f docker/Dockerfile.cuda12 .
```
### B. Building Docker Image with CUDA 11 and Python 3.12

```bash
docker build -t ctrlaltdiffuse-cuda11 -f docker/Dockerfile.cuda11 .
```
## Running Containers in Different Modes

### A. Running in Interactive Mode

Use the interactive mode to run the container and manually execute commands inside it.

#### CUDA 12 and Python 3.12

```bash
docker run --gpus all -it ctrlaltdiffuse-cuda12
```

#### CUDA 11 and Python 3.12

```bash
docker run --gpus all -it ctrlaltdiffuse-cuda11
```

### B. Running in Training Mode

To start training directly, use the following commands to run the container.

#### CUDA 12 and Python 3.12

```bash
docker run --gpus all ctrlaltdiffuse-cuda12 python src/diffuse_trainer.py --dataset-type celebs --dataset-path ./dataset --output ./output
```

#### CUDA 11 and Python 3.12

```bash
docker run --gpus all ctrlaltdiffuse-cuda11 python src/diffuse_trainer.py --dataset-type celebs --dataset-path ./dataset --output ./output
```

### C. Running in Image Generation Mode

To run the container for image generation using a trained model:

#### CUDA 12 and Python 3.12

```bash
docker run --gpus all ctrlaltdiffuse-cuda12 python src/diffuse_generator.py --checkpoints ./output/checkpoint.pth --image_dimensions 256 256
```

#### CUDA 11 and Python 3.12

```bash
docker run --gpus all ctrlaltdiffuse-cuda11 python src/diffuse_generator.py --checkpoints ./output/checkpoint.pth --image_dimensions 256 256
```
# Quickstart
If you have not installed the repository as a package, you can call still run the python scripts directly.\
For example instead of ```diffuse-train -h```, run ```python diffuse_train.py -h``` when in the correct directory.
## Training
Use the following command to train:
```
diffuse-train --dataset-type [celebs or flowers] --dataset-path ./dataset --output ./output
```
To see all arguments, call ```diffuse-train -h``` or see the following list:
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for diffuse-train and diffuse_train.py</span></summary>

  ### --dataset-type
  Type of dataset to be used (e.g., "celebs", "flowers"). "celebs" by default.
  ### --dataset-path
  Path to the dataset directory. "./datasets" by default
  ### --checkpoints
  Path to load checkpoint of trained model; None if not used.
  ### --output
  Path to store the checkpoint of the trained model. "./output" by default.
  ### --image_dimensions
  Input image dimensions (height, width). Default: (256, 256)
  ### --batch_size
  Number of samples per batch. Default: 32
  ### --epochs
  Number of training epochs. Default: 10
  ### --learning_rate
  Learning rate for the optimizer. Default: 0.0001
  ### --optimizer
  Optimizer type (e.g., "adam", "sgd"). "adam" by default.

</details>

## Image generation
Use the following command to generate an image using a trained model:
```
diffuse-generate 
```
To see all arguments, call ```diffuse-generate -h``` or see the following list:
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for diffuse-generate and diffuse_generate.py</span></summary>

### --checkpoints
Path to the trained model
### --image_dimensions
The dimension of the generated image. Default: (256, 256)
</details>