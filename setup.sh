#!/bin/bash

uv sync --all-extras --group dev --preview-features extra-build-dependencies
source .venv/bin/activate

pybind11-stubgen geometry_cuda.ba -o /workspace/geometry/src
pybind11-stubgen geometry_cuda.proj -o /workspace/geometry/src
pybind11-stubgen geometry_cuda.lie -o /workspace/geometry/src
pybind11-stubgen neural_cuda.corr -o /workspace/neural/src
