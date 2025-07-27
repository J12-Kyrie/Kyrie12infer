# Use Ubuntu base image with Python
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies and CUDA toolkit
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-1 \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.0-1_all.deb

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (with retry mechanism)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --retries 10 --timeout 300 \
    torch>=2.4.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 || \
    (echo "Retrying PyTorch installation with different approach..." && \
     pip install --no-cache-dir --retries 5 --timeout 600 \
     torch torchvision torchaudio \
     --index-url https://download.pytorch.org/whl/cu121)

# Install other dependencies including flash-attn (with retry and error handling)
RUN pip install --no-cache-dir \
    ninja \
    packaging \
    wheel && \
    echo "Installing flash-attn with CUDA support..." && \
    (pip install --no-cache-dir --retries 5 --timeout 600 \
     flash-attn --no-build-isolation || \
     echo "Warning: flash-attn installation failed, continuing without it...") && \
    pip install --no-cache-dir --retries 3 --timeout 300 \
    transformers>=4.51.0 \
    xxhash \
    tqdm

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create a non-root user
RUN useradd -m -u 1000 nanovllm && chown -R nanovllm:nanovllm /app
USER nanovllm

# Expose port for potential API server
EXPOSE 8000

# Default command
CMD ["python", "bench.py"]