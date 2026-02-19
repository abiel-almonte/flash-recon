import torch

from geometry import (
    Pose,
    Intrinsics,
    pose_to_matrix,
    pose_inv,
    get_meshgrid,
)


def backproject(
    frame: torch.Tensor,
    pose: Pose,
    disps: torch.Tensor,
    vmask: torch.Tensor,
    intrinsics: Intrinsics,
    stride: int = 2,
):
    """Backproject depth map to 3D Gaussians.

    Args:
        disps: [H, W] disparity map
        vmask: [H, W] valid depth mask
        pose: Pose object (w2c)
        frame: [H, W, 3] RGB image
        intrinsics: camera intrinsics
        stride: subsampling stride

    Returns:
        (world, colors, scales, viewmat, pixel_uv) or None if no valid points
    """
    c2w = pose_to_matrix(pose_inv(pose)).squeeze(0)  # [4, 4]
    device = disps.device

    z = 1.0 / disps.clamp(1e-5)

    vmask = vmask[::stride, ::stride]
    z = z[::stride, ::stride]

    v, u = get_meshgrid(*z.shape, device, torch.float)
    u = u * stride
    v = v * stride

    mask = vmask & (z > 0.01) & (z < 10.0) & torch.isfinite(z)
    u, v, z = u[mask], v[mask], z[mask]

    if u.shape[0] == 0:
        return None

    fx, fy, cx, cy = tuple(intrinsics)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    cam = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)

    world = (c2w @ cam.reshape(-1, 4).T).T[:, :3]

    H, W = frame.shape[:2]
    vi = v.long().clamp(0, H - 1)
    ui = u.long().clamp(0, W - 1)
    colors = frame[vi, ui]

    pixel_size = z / fx * stride
    scales = pixel_size.unsqueeze(-1).expand(-1, 3).log()  # store in log space

    pixel_uv = torch.stack([ui, vi], dim=-1)

    viewmat = torch.inverse(c2w).unsqueeze(0)

    return world, colors, scales, viewmat, pixel_uv
