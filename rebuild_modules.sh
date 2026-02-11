#!/bin/bash

export LD_LIBRARY_PATH="/workspace/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"

cd /workspace/geometry \
&& rm -rf build *.egg-info \
&& uv pip install -e . --no-build-isolation \
&& pybind11-stubgen geometry_cuda.ba -o /workspace/geometry/src \
&& pybind11-stubgen geometry_cuda.proj -o /workspace/geometry/src \
&& pybind11-stubgen geometry_cuda.lie -o /workspace/geometry/src \
&& black .

cd /workspace/neural \
&& rm -rf build *.egg-info \
&& uv pip install -e . --no-build-isolation \
&& pybind11-stubgen neural_cuda.corr -o /workspace/neural/src \
&& black .
