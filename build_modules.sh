#!/bin/bash

pip install --no-build-isolation -e /workspace/pose_utils/ \
&& pybind11-stubgen lie_ops_cuda -o /workspace/pose_utils/src


pip install --no-build-isolation -e /workspace/projective_utils/ \
&& pybind11-stubgen projective_ops_cuda -o /workspace/projective_utils/src
