import functools
import torch

from structs import Pose, Intrinsics

from geometry_cuda.proj import (
    proj_cuda,
    proj_jac_cuda,
    fused_projective_cuda,
    fused_projective_jac_cuda,
    fused_induced_flow_cuda,
    fused_depth_filter_cuda,
)

MIN_DEPTH = 0.2


@functools.lru_cache(maxsize=None)
def get_meshgrid(ht, wd, device, dtype):
    return torch.meshgrid(
        torch.arange(ht, device=device, dtype=dtype),
        torch.arange(wd, device=device, dtype=dtype),
    )


@functools.lru_cache(maxsize=None)
def get_camera_grid(ht, wd, fx, fy, cx, cy, device, dtype):
    y, x = get_meshgrid(ht, wd, device, dtype)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    i = torch.ones_like(x)
    return torch.stack([X, Y, i], dim=-1)


def iproj(disps, intrinsics: Intrinsics):
    """Pinhole camera inverse projection."""
    # Handle both [..., H, W] and [N, H, W] shapes
    if len(disps.shape) >= 3:
        ht, wd = disps.shape[-2:]
    else:
        raise ValueError(f"Expected at least 3D tensor, got shape {disps.shape}")
    fx, fy, cx, cy = intrinsics

    base_grid = get_camera_grid(ht, wd, fx, fy, cx, cy, disps.device, disps.dtype)

    pts = torch.empty((*disps.shape, 4), device=disps.device, dtype=disps.dtype)
    pts[..., :3] = base_grid
    pts[..., 3] = disps

    return pts


def proj(Xs: torch.Tensor, intrinsics: Intrinsics, return_depth=False):
    fx, fy, cx, cy = intrinsics
    Xs_flat = Xs.reshape(-1, 4)

    last_dim = 2
    if return_depth:
        last_dim += 1

    coords = proj_cuda(Xs_flat, fx, fy, cx, cy, last_dim)
    return coords.reshape(*Xs.shape[:-1], last_dim)


@functools.lru_cache(maxsize=None)
def iproj_jac(disps_shape, device, dtype):
    """Jacobian of inverse projection wrt disparity/depth."""
    J = torch.zeros((*disps_shape, 4), device=device, dtype=dtype)
    J[..., -1] = 1.0
    return J


def proj_jac(Xs: torch.Tensor, intrinsics: Intrinsics):
    fx, fy, cx, cy = intrinsics
    Xs_flat = Xs.reshape(-1, 4)
    jac = proj_jac_cuda(Xs_flat, fx, fy, cx, cy)

    return jac.reshape(*Xs.shape[:-1], 2, 4)


def projective_transform_fused(
    poses: Pose,
    depths: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
):
    """Fused iproj->SE3->proj implementation.

    Returns coords[edges,H,W,2], valid[edges,H,W,1].
    """
    coords, valid = fused_projective_cuda(
        poses.t, poses.q, depths, intrinsics.as_tensor, ii, jj
    )
    return coords, valid


def projective_transform_jac_fused(
    poses: Pose,
    depths: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
):
    coords, valid, Ji, Jj, Jz = fused_projective_jac_cuda(
        poses.t, poses.q, depths, intrinsics.as_tensor, ii, jj
    )
    return coords, valid, Ji, Jj, Jz


def induced_flow_fused(
    poses: Pose,
    depths: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
):
    """Fused iproj->SE3->proj implementation computing optical flow

    Returns coords[edges,H,W,2], valid[edges,H,W,1].
    """
    coords, valid = fused_induced_flow_cuda(
        poses.t, poses.q, depths, intrinsics.as_tensor, ii, jj
    )
    return coords, valid


def depth_filter_fused(
    poses: Pose,
    depths: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    thresh: torch.Tensor,
):
    count = fused_depth_filter_cuda(
        poses.t, poses.q, depths, intrinsics.as_tensor, ii, thresh
    )
    return count
