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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install PyTorch with dynamic CUDA version support, without caching
ARG TORCH_CUDA_VERSION
RUN if [ "$CUDA_VERSION" = "12.1.0" ]; then \
        TORCH_CUDA_VERSION=cu121; \
    else \
        TORCH_CUDA_VERSION=cu118; \
    fi \
    && pip3 --no-cache-dir install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_CUDA_VERSION}

# Create the necessary cache directory for gdown and set ownership
RUN mkdir -p /home/appuser/.cache/gdown && chown -R appuser:appuser /home/appuser/.cache


# Set working directory
WORKDIR /workspace

# Install project dependencies without caching
COPY requirements.txt /workspace/
RUN if [ -f "requirements.txt" ]; then pip3 --no-cache-dir install -r requirements.txt; fi

# Copy the project files into the container
COPY .. /workspace
# Change ownership of the workspace directory to appuser
RUN chown -R appuser:appuser /workspace

# Install the project in editable mode
RUN pip3 install --no-cache-dir -e . && rm -rf /tmp/*

# Change to non-root user
USER appuser

# Set entry point for either interactive shell or commands (training/generation)
ENTRYPOINT [ "/bin/bash", "-c" ]
CMD ["bash"]
