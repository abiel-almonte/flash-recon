import torch

from .structs import Pose, Intrinsics
from .utils import (
    projective_transform_jac_fused,
    projective_transform_fused,
    induced_flow_fused,
    depth_filter_fused,
    frame_distance_fused,
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


def depth_filter(
    poses: Pose,
    disps: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    thresh: torch.Tensor,
):
    return depth_filter_fused(poses, disps, intrinsics, ii, thresh)


def compute_distance(
    poses: Pose,
    disps: torch.Tensor,
    intrinsics: Intrinsics,
    ii=None,
    jj=None,
    t0: int = None,
    t1: int = None,
    t0_loop: int = None,
    beta: float = 0.3,
    bidirectional: bool = False,
):
    """Compute reprojection-based frame distance"""
    device = poses.t.device
    matrix_mode = t0 is not None and t1 is not None

    if matrix_mode:
        if t0_loop is None:
            t0_loop = t0
        rows = torch.arange(t0_loop, t1, device=device)
        cols = torch.arange(t0, t1, device=device)
        ii_grid, jj_grid = torch.meshgrid(rows, cols, indexing="ij")
        ii_flat = ii_grid.reshape(-1)
        jj_flat = jj_grid.reshape(-1)
    else:
        if isinstance(ii, list):
            ii = torch.as_tensor(ii, dtype=torch.long, device=device)
        if isinstance(jj, list):
            jj = torch.as_tensor(jj, dtype=torch.long, device=device)
        ii_flat = ii
        jj_flat = jj

    dist = frame_distance_fused(poses, disps, intrinsics, ii_flat, jj_flat, beta)

    if bidirectional:
        dist_rev = frame_distance_fused(poses, disps, intrinsics, jj_flat, ii_flat, beta)
        dist = 0.5 * (dist + dist_rev)

    if matrix_mode:
        n_rows = t1 - t0_loop
        n_cols = t1 - t0
        dist = dist.reshape(n_rows, n_cols)

    return dist