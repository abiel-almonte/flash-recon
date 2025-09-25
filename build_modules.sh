#!/bin/bash

pip install --no-build-isolation -e /workspace/pose_utils/ \
&& pybind11-stubgen lie_ops_cuda -o /workspace/pose_utils/src


pip install --no-build-isolation -e /workspace/projective_utils/ \
&& pybind11-stubgen projective_ops_cuda -o /workspace/projective_utils/src


pip install --no-build-isolation -e /workspace/ba_utils/ \
&& pybind11-stubgen ba_ops_cuda -o /workspace/ba_utils/src


pip install --no-build-isolation -e /workspace/vision_rt/ \
&& pybind11-stubgen vision_rt_cuda -o /workspace/vision_rt/src
