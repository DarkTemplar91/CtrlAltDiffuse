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
### A. Clone repository
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

### B. Install package directly
After setting up the base environment by installing the correct version of PyTorch, you can also install the library
directly as a pip package:
```
pip install git+https://github.com/DarkTemplar91/CtrlAltDiffuse.git
```

## Docker

### A. Using Pre-built Docker Images in Production

In production, you don't need to build the Docker images manually. Instead, you can pull the pre-built images from the GitHub Container Registry (GHCR) and run them with the following commands:

For CUDA 12.1:
```bash
docker pull ghcr.io/darktemplar91/ctrlaltdiffuse-cuda12:latest
docker run -p 5005:5005 --gpus all --shm-size=16g -v /path/to/data:/workspace/data -v /path/to/checkpoints:/workspace/checkpoints -it ghcr.io/darktemplar91/ctrlaltdiffuse-cuda12:latest
```
For CUDA 11.8:
```bash
docker pull ghcr.io/darktemplar91/ctrlaltdiffuse-cuda11:latest
docker run -p 5005:5005 --gpus all --shm-size=16g -v /path/to/data:/workspace/data -v /path/to/checkpoints:/workspace/checkpoints -it ghcr.io/darktemplar91/ctrlaltdiffuse-cuda11:latest
```
### B. Building Docker Images with CUDA Support

You can build Docker images with either CUDA 11.8 or CUDA 12.1 support. The default version is CUDA 11.8, but you can specify CUDA 12.1 by using a build argument.

#### Build CUDA 11.8 Image (default)

```bash
docker build -t ctrlaltdiffuse-cuda:cuda11 --build-arg CUDA_VERSION=11.8 -f docker/Dockerfile .
```

#### Build CUDA 12.1 Image

```bash
docker build -t ctrlaltdiffuse-cuda:cuda12 --build-arg CUDA_VERSION=12.1 -f docker/Dockerfile .
```

The docker images are also available in the "Packages" section in GitHub.

## Quickstart: Running the Container in Different Modes

### A. Starting the Docker Container

Use the following command to start the Docker container with GPU support, increased memory allocation, and mounted data/checkpoints.\
This command will open an interactive terminal where you can use the ```diffuse-train``` and ```diffuse-generate```
commands directly.

```bash
docker run -p 5005:5005 --gpus all --shm-size=16g -v /path/to/data:/workspace/data -v /path/to/checkpoints:/workspace/checkpoints -it ctrlaltdiffuse-cuda:cuda11
```

- **`--shm-size=16g`**: Increases the shared memory size to 16 GB, which may be necessary for larger models.
- **`-v /path/to/data:/workspace/data`** and **`-v /path/to/checkpoints:/workspace/checkpoints`**: Mounts the data and checkpoints folders from the host system into the container's `/workspace` directory.


To use CUDA 12.1, replace `ctrlaltdiffuse-cuda:cuda11` with `ctrlaltdiffuse-cuda:cuda12` in the command above.

### B. Running in Training Mode

To start training directly within the container, use the following command:

```bash
docker run -p 5005:5005 --gpus all --shm-size=16g -v /path/to/data:/workspace/data -v /path/to/checkpoints:/workspace/checkpoints -it ctrlaltdiffuse-cuda:cuda11 diffuse-train --dataset-type celebs --dataset-path ./data --output ./output
```

For CUDA 12.1, use:

```bash
docker run -p 5005:5005 --gpus all --shm-size=16g -v /path/to/data:/workspace/data -v /path/to/checkpoints:/workspace/checkpoints -it ctrlaltdiffuse-cuda:cuda12 diffuse-train --dataset-type celebs --dataset-path ./data --output ./output
```

### C. Running in Image Generation Mode

To run the container for image generation using a trained model, use the following command:

```bash
docker run -p 5005:5005 --gpus all --shm-size=16g -v /path/to/data:/workspace/data -v /path/to/checkpoints:/workspace/checkpoints -it ctrlaltdiffuse-cuda:cuda11 diffuse-generate --checkpoints ./checkpoints/checkpoint.pth --image_resolution 256
```

For CUDA 12.1, use:

```bash
docker run -p 5005:5005 --gpus all --shm-size=16g -v /path/to/data:/workspace/data -v /path/to/checkpoints:/workspace/checkpoints -it ctrlaltdiffuse-cuda:cuda12 diffuse-generate --checkpoints ./checkpoints/checkpoint.pth --image_resolution 256
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
  ### --image_resolution
  Input image resolution. This will be the smaller edge of the image. Default: 256
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
### --image_resolution
The resolution of the generated image. Default: 256
</details>

# Data analysis
We analyzed the input images in the ```data_analyse.ipynb``` Jupyter Notebook.\
In the notebook, we did the following:

- Used **PyTorch** to extract features from the selected datasets using a pre-trained VGG16 model.
- Perform dimensionality reduction using **PCA**.
- Apply **K-Means** clustering to group similar images.
- Use **t-SNE** for visualization.
- Create an interactive plot within the Jupyter Notebook where clicking on a point displays the corresponding image.
- Create barchart for Mean RGB values
- Create histogram for Color intensity

# Reference Model and Benchmarking

1. **Download the Reference Model**  
   You can download the reference model from the following link:  
   [Download Flowers Reference Model](https://bmeedu-my.sharepoint.com/:u:/g/personal/somodi_istvan_edu_bme_hu/EZko3h8MZEhEjZ3oVcmg2cYBv6-ZgrymSShTXZncQdLJlg).

   You can download the reference model from the following link:  
   [Download CelebA Model](https://drive.google.com/file/d/1ZcAxwv1JerjY-2EzJ_ybUCYPHXUESgHi).

1.w **Download the Last Model**  

   You can download the  model from the following link:  
   [Download Flowers Model](https://drive.google.com/file/d/1ME3R0VSQCE6Ve8W6F3aZTLVpC65jNV70).

   [Download CelebA Model](https://drive.google.com/file/d/1ZcAxwv1JerjY-2EzJ_ybUCYPHXUESgHi).


2. **Benchmarking Notebook**  
   To evaluate the Reference model, we used the ```benchmark_FID_IS_KID.ipynb``` notebook. This notebook calculates three key metrics:
   - **FID (Frechet Inception Distance)**: Measures the distance between real and generated images in feature space. Lower FID scores indicate closer similarity and better-quality generated images.
   - **IS (Inception Score)**: Evaluates image diversity and quality based on the likelihood of recognizable classes in generated images. Higher IS scores are better.
   - **KID (Kernel Inception Distance)**: Similar to FID but generally more robust for smaller sample sizes, providing another perspective on similarity between real and generated images.

3. **Reference Model Scores**  
   The benchmark results for the reference model are as follows:
   - **FID Score**: 349.2197
   - **Inception Score**: Mean = 1.4570, Std = 0.2036
   - **KID Score**: 0.2482

These scores provide a baseline for evaluating the model's quality in generating realistic and diverse images.

**How to run the pipeline?**
   - Use `reference_generator.py` to run the image generation pipeline. This script generates images based on a trained model checkpoint.
   - Example command:
     ```bash
     python reference_generator.py
     ```

2. **How to train the models?**
   - Training is handled by the `reference_train.py` script. Adjust hyperparameters such as batch size, learning rate, and epochs directly in the script or via command-line arguments.
   - Example command:
     ```bash
     python reference_train.py 
     ```

3. **How to evaluate the models?**
   - Evaluation metrics, including FID, IS, and KID, are available in the `benchmark_<dataser>_FID_IS_KID.ipynb` notebook. Run each cell to compute these metrics, which assess the quality and diversity of the generated images.

# Diffusion Model Image Generation GUI

## Accessing the GUI
The GUI provides an intuitive interface to generate images using pre-trained diffusion models. Depending on your deployment method, follow the instructions below:

### Docker Deployment
1. Build or Pull the image, run the Docker container as described in the "Docker" section.
2. Open your browser and navigate to [http://localhost:5005](http://localhost:5005).

### Local Deployment
1. Run the Flask application locally:
   ```bash
   pip install -e .
   cd app
   python app.py 
   ```
Note: Its important that you start the app.py from root of repo.
2. Open your browser and navigate to [http://localhost:5005](http://localhost:5005).

---

## GUI Features

### Main Features
- **Dataset**: Set default path of checkpoint when we are using docker.
- **Checkpoint Selection**: Use the "Browse" button to select a pre-trained model checkpoint file from your local machine.
- **Number of Steps**: Define the number of diffusion steps. Higher steps generally improve image quality but take longer.
- **Random Seed**: Set a random seed for reproducibility or leave it blank for random generation.
- **Number of Images**: Specify the number of images to generate in a single run.
- **Generate Button**: Click the "Generate Images" button to create images based on the selected parameters.
- **Preview of Generated Images**: View the generated images directly in the browser.

### Additional Notes
- The generated images are saved in the `app/static/generated_images/` directory within the container or local setup.
- Default checkpoints are preloaded for CelebA and Flowers datasets, but you can upload custom models as needed.

---

## Example Workflow

### Steps to Generate Images
1. **Checkpoint Selection**:
   - Use the "Checkpoint" field to upload the desired checkpoint file.
   - If no file is selected, the default CelebA or Flowers checkpoint will be used.(The Docker image contains both checkpoint)

2. **Set Parameters**:
   - **Number of Steps**: Choose the number of diffusion steps (e.g., 1000).
   - **Random Seed**: Set a seed for reproducibility or leave it blank for random behavior.
   - **Number of Images**: Specify how many images you want to generate.

3. **Generate Images**:
   - Click the "Generate Images" button.
   - Wait for the images to appear below the form.

4. **Inspect and Save**:
   - Right-click on any generated image to save it.
   - Alternatively, find all images in the `app/static/generated_images/` directory.

---

## GUI Preview
![GUI Preview](plots/gui_preview_image.png)

*Example of the GUI interface for generating images.*


# Administrative Informations

## Team members

- Somodi István - IXH8RO
- Szász Erik - B7RBBU

## Related works
- https://arxiv.org/abs/2006.11239
- https://arxiv.org/abs/2010.02502
- https://keras.io/examples/generative/ddim/
- https://github.com/ermongroup/ddim
- https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.TSNE.html
- https://www.datacamp.com/tutorial/introduction-t-sne