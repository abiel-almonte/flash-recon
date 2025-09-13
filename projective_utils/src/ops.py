import functools
import torch

from pose_utils import (
    Pose,
    Intrinsics,
    pose_inv,
    pose_mul,
    transform_points_by_pose,
    points_to_pose_jacobian,
    pose_adjointT,
)

from projective_ops_cuda import proj_cuda, proj_jac_cuda

MIN_DEPTH = 0.01


@functools.lru_cache(maxsize=None)
def get_camera_grid(ht, wd, fx, fy, cx, cy, device, dtype):
    y, x = torch.meshgrid(
        torch.arange(ht, device=device, dtype=dtype),
        torch.arange(wd, device=device, dtype=dtype),
        indexing="ij",
    )
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


def projective_transform(
    poses: Pose,
    depths: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
    jacobian: bool = False,
    return_depth: bool = False,
):
    """Map points from frame ii to frame jj."""

    # Handle indexing properly - depths[ii] gives [len(ii), H, W]
    depths_ii = depths[ii]  # [len(ii), H, W]
    X0 = iproj(depths_ii, intrinsics).contiguous()  # Ensure contiguous for CUDA
    Gij = pose_mul(poses[jj], pose_inv(poses[ii]))

    mask = ii == jj
    if mask.any():
        n_same = int(mask.sum())
        device, dtype = Gij.q.device, Gij.q.dtype

        identity_t = torch.zeros((n_same, 3), device=device, dtype=dtype)
        identity_t[:, 0] = -0.1
        identity_q = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype).expand(
            n_same, -1
        )

        Gij.t[mask] = identity_t
        Gij.q[mask] = identity_q

    X1 = transform_points_by_pose(Gij, X0)

    x1 = proj(X1, intrinsics, return_depth)

    valid = ((X1[..., 2] > MIN_DEPTH) & (X0[..., 2] > MIN_DEPTH)).float().unsqueeze(-1)

    if not jacobian:
        return x1, valid

    Jz = iproj_jac(depths.shape, depths.device, depths.dtype)  # wrt depths
    Jp = proj_jac(X1, intrinsics)  # wrt projection
    Ja = points_to_pose_jacobian(X1)  # wrt pose_j

    Jj = torch.matmul(Jp, Ja)
    Ji = -pose_adjointT(Gij, Jj)
    Jz_trans = transform_points_by_pose(Gij, Jz)
    Jz = torch.matmul(Jp, Jz_trans.unsqueeze(-1))

    return x1, valid, (Ji, Jj, Jz)
