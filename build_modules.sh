#!/bin/bash

pip install --no-build-isolation -e /workspace/geometry/ \
&& pybind11-stubgen geometry_cuda.ba -o /workspace/geometry/src \
&& pybind11-stubgen geometry_cuda.proj -o /workspace/geometry/src \
&& pybind11-stubgen geometry_cuda.lie -o /workspace/geometry/src

pip install --no-build-isolation -e /workspace/neural/ \
&& pybind11-stubgen neural_cuda.corr -o /workspace/neural/src

