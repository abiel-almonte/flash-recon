import torch

from geometry import (
    Pose,
    pose_to_matrix,
    pose_inv,
    pose_mul,
    quat_multiply_cuda,
)


def deform_gaussians(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    pixel_uv: torch.Tensor,
    kf_ids: torch.Tensor,
    old_poses: Pose,
    new_poses: Pose,
    old_disps: torch.Tensor,
    new_disps: torch.Tensor,
):
    """Deform all gaussians in one batched pass (Eq 12).

    Args:
        means: [N, 3] gaussian positions
        scales: [N, 3] log-space scales
        quats: [N, 4] quaternions
        pixel_uv: [N, 2] pixel coordinates (u, v)
        kf_ids: [N] keyframe index per gaussian
        old_poses: [K] old w2c poses
        new_poses: [K] new w2c poses
        old_disps: [K, H, W] old disparity maps
        new_disps: [K, H, W] new disparity maps

    Returns:
        (new_means, new_scales, new_quats)
    """
    # per-gaussian depth from disparity maps: [N]
    old_depth = 1.0 / old_disps[kf_ids, pixel_uv[:, 1], pixel_uv[:, 0]].clamp(1e-5)
    new_depth = 1.0 / new_disps[kf_ids, pixel_uv[:, 1], pixel_uv[:, 0]].clamp(1e-5)

    # per-keyframe transforms: [K, 4, 4]
    old_c2w = pose_to_matrix(pose_inv(old_poses))
    new_c2w = pose_to_matrix(pose_inv(new_poses))
    old_w2c = torch.inverse(old_c2w)

    # index to per-gaussian: [N, 4, 4]
    ow = old_w2c[kf_ids]
    nc = new_c2w[kf_ids]

    # to camera space: [N, 3]
    cam_pts = torch.einsum("nij,nj->ni", ow[:, :3, :3], means) + ow[:, :3, 3]
    cam_z = cam_pts[:, 2:3]

    scale_ratio = 1.0 + (new_depth - old_depth).unsqueeze(-1) / cam_z

    # scale and transform back to world: [N, 3]
    cam_pts_scaled = scale_ratio * cam_pts
    new_means = torch.einsum("nij,nj->ni", nc[:, :3, :3], cam_pts_scaled) + nc[:, :3, 3]

    new_scales = scales + scale_ratio.clamp(min=1e-6).log()

    # per-keyframe relative rotation, indexed to per-gaussian
    rel = pose_mul(pose_inv(new_poses), old_poses)
    new_quats = quat_multiply_cuda(rel.q[kf_ids], quats)

    return new_means, new_scales, new_quats
