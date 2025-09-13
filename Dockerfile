FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"

RUN apt update \
&& apt install -y python3 python3-pip python-is-python3 python3-full v4l-utils wget unzip git

RUN python3 -m venv /opt/.venv
ENV PATH="/opt/.venv/bin:$PATH"

RUN pip install --upgrade pip \
&& pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
&& pip install pybind11 pybind11-stubgen black \
&& pip install --no-build-isolation git+https://github.com/princeton-vl/lietorch.git

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/.venv/lib/python3.12/site-packages/torch/lib

RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.8.0%2Bcu128.zip \
&& unzip libtorch.zip -d /opt \
&& rm libtorch.zip

WORKDIR /workspace
