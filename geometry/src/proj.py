import torch

from structs import Pose, Intrinsics
from utils import (
    projective_transform_jac_fused,
    projective_transform_fused,
    induced_flow_fused,
)


def projective_transform(
    poses: Pose,
    depths: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
    jacobian: bool = False,
):
    """Map points from ii->jj using fused CUDA paths. Computes jacobians if requested."""
    if jacobian:
        return projective_transform_jac_fused(poses, depths, intrinsics, ii, jj)
    return projective_transform_fused(poses, depths, intrinsics, ii, jj)


def induced_flow(
    poses: Pose,
    disps: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
):
    """optical flow induced by camera motion"""
    return induced_flow_fused(poses, disps, intrinsics, ii, jj)
