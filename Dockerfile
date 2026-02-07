FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"

RUN apt update && apt install -y \
    python3 \
    python-is-python3 \
    python3-pip \
    python3-full \
    v4l-utils \
    wget \
    curl \
    unzip \
    git \ 
    libgl1 \ 
    libglib2.0-0

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN git config --global --add safe.directory /workspace
ENV UV_LINK_MODE=copy

ENV PATH="/root/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/workspace/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"

RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.10.0%2Bcu128.zip \
    && unzip libtorch.zip -d /opt \
    && rm libtorch.zip

WORKDIR /workspace
