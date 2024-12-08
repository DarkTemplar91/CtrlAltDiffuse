# Use a build argument to specify CUDA version
ARG CUDA_VERSION=11.8

# Use a minimal CUDA base image, dynamically setting the CUDA version
FROM nvidia/cuda:${CUDA_VERSION}.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Budapest

# Install system dependencies (Python is assumed to be installed in the base image)
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    build-essential \
    python3.11 \
    python3-pip \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install gdown for Google Drive file downloads
RUN pip3 install --no-cache-dir gdown

# Create necessary directories
RUN mkdir -p /workspace/outputs/celeba /workspace/outputs/flowers

# Download models and place them in the respective directories
RUN gdown --id 1ZcAxwv1JerjY-2EzJ_ybUCYPHXUESgHi -O /workspace/outputs/celeba/diffusion_model_celeba.ckpt \
    && gdown --id 1ME3R0VSQCE6Ve8W6F3aZTLVpC65jNV70 -O /workspace/outputs/flowers/diffusion_model_flowers.ckpt

# Set working directory
WORKDIR /workspace

# Copy the project files into the container
COPY .. /workspace

# Create a non-root user and change ownership of the workspace directory
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /workspace

# Create Matplotlib configuration directory and set ownership
RUN mkdir -p /workspace/matplotlib_config \
    && chown -R appuser:appuser /workspace/matplotlib_config

# Install project dependencies without caching
COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt \
    && pip3 install --no-cache-dir -e . \
    && rm -rf /tmp/*

# Expose Flask port
EXPOSE 5005

ENV FLASK_APP=app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5005

# Change to non-root user
USER appuser

# Set entry point for Flask application
ENTRYPOINT ["python3", "app/app.py"]
