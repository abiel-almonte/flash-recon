import torch

from pose_utils import Pose, Intrinsics
from ba_ops_cuda import (
    fused_projective_transform_with_reduction_cuda,
    fused_depth_jacobians_cuda,
)


def projective_transform_with_reduction_fused(
    poses: Pose,
    depths: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
):
    Ck, wk = fused_projective_transform_with_reduction_cuda(
        poses.t, poses.q, depths, intrinsics.as_tensor, ii, jj, target, weight
    )
    return Ck, wk


def depth_jacobians_fused(
    disps: torch.Tensor,
    mono: torch.Tensor,
    depth_mask: torch.Tensor,
    scales: torch.Tensor,
    shifts: torch.Tensor,
    ignore: torch.Tensor,
    alpha: float,
):
    Jwq, Jd, Rd = fused_depth_jacobians_cuda(
        disps, mono, depth_mask, scales, shifts, ignore, alpha
    )
    return Jwq, Jd, Rd


def projective_jacobians(
    poses: Pose,
    depths: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
):
    return projective_transform_with_reduction_fused(
        poses, depths, intrinsics, ii, jj, target, weight
    )


def depth_jacobians(
    disps: torch.Tensor,
    mono_disps: torch.Tensor,
    scales: torch.Tensor,
    shifts: torch.Tensor,
    depth_mask: torch.Tensor,
    frame_bound: int,
    alpha: float,
    indices: torch.Tensor,
):
    disps_i = disps[indices].contiguous()
    mono_disps_i = mono_disps[indices].contiguous()
    depth_mask_i = depth_mask[indices].contiguous()
    scales_i = scales[indices].contiguous()
    shifts_i = shifts[indices].contiguous()
    ignore_i = indices < frame_bound

    return depth_jacobians_fused(
        disps_i, mono_disps_i, depth_mask_i, scales_i, shifts_i, ignore_i, alpha
    )


def assemble_scale_shift_sys(
    scale_shift_jac: torch.Tensor,
    depth_jac: torch.Tensor,
    depth_residual: torch.Tensor,
    proj_depth_diag: torch.Tensor,
    proj_depth_residual: torch.Tensor,
    eta: torch.Tensor,
    kk: torch.Tensor,
):
    M, hw, _ = scale_shift_jac.shape
    idx = torch.arange(M, device=scale_shift_jac.device)

    scale_shift_hessian = torch.zeros(
        (M, M, 2, 2), device=scale_shift_jac.device, dtype=scale_shift_jac.dtype
    )
    cross_term = torch.zeros(
        (M, M, 2, hw), device=scale_shift_jac.device, dtype=scale_shift_jac.dtype
    )

    proj_residual_scatter = torch.zeros(
        (M, hw), device=scale_shift_jac.device, dtype=scale_shift_jac.dtype
    )
    proj_diag_scatter = torch.zeros(
        (M, hw), device=scale_shift_jac.device, dtype=scale_shift_jac.dtype
    )

    scale_shift_jac_depth_residual = torch.zeros(
        (M, hw, 3), device=scale_shift_jac.device, dtype=scale_shift_jac.dtype
    )
    scale_shift_jac_depth_residual[..., :2] = scale_shift_jac
    scale_shift_jac_depth_residual[..., 2:3] = depth_residual.unsqueeze(-1)

    scale_shift_jacT = scale_shift_jac.transpose(-2, -1)  # [M, 2, hw]
    prod = torch.bmm(scale_shift_jacT, scale_shift_jac_depth_residual)  # [M, 2, 3]

    hessian_update, rhs_update = prod.split([2, 1], dim=-1)
    scale_shift_hessian[idx, idx] = hessian_update  # [M, M, 2, 2]
    scale_shift_rhs = -rhs_update.squeeze(-1)  # [M, 2]
    cross_term[idx, idx] = scale_shift_jacT * depth_jac.unsqueeze(1)  # [M, M, 2, hw]

    proj_residual_scatter.index_add_(0, kk, proj_depth_residual)  # [M, hw]
    proj_diag_scatter.index_add_(0, kk, proj_depth_diag)  # [M, hw]

    depth_diag = (
        proj_diag_scatter + depth_jac.pow(2) + eta.view(*proj_depth_diag.shape)
    )  # [M, hw]
    depth_rhs = -(proj_residual_scatter + depth_jac * depth_residual)  # [M, hw]

    return scale_shift_hessian, scale_shift_rhs, cross_term, depth_diag, depth_rhs


def schur_solve(
    scale_shift_hessian: torch.Tensor,
    cross_term: torch.Tensor,
    depth_diag: torch.Tensor,
    scale_shift_rhs: torch.Tensor,
    depth_rhs: torch.Tensor,
    ep: float = 0.1,
    lm: float = 0.0001,
):
    """solve using shur complement"""

    M, _, d, hw = cross_term.shape
    scale_shift_hessian = scale_shift_hessian.permute(0, 2, 1, 3).reshape(
        M * d, M * d
    )  # [M*d, M*d]
    cross_term = cross_term.permute(0, 2, 1, 3).reshape(M * d, M * hw)  # [M*d, M*hw]
    inv_depth_diag = (1.0 / depth_diag).view(M * hw, 1)
    scale_shift_rhs = scale_shift_rhs.reshape(M * d, 1)
    depth_rhs = depth_rhs.reshape(M * hw, 1)

    scale_shift_hessian.diagonal().mul_(1 + lm).add_(ep)

    cross_termT = cross_term.transpose(-2, -1)  # [M*hw, M*d]
    rhs = inv_depth_diag * torch.cat(
        [cross_termT, depth_rhs], dim=-1
    )  # [M*hw, M*d + 1]
    prod = torch.matmul(cross_term, rhs)  # [M*d, M*d + 1]

    scale_shift_update, rhs_update = prod.split([M * d, 1], dim=-1)
    scale_shift_hessian.sub_(scale_shift_update)  # [M*d, M*d]
    scale_shift_rhs.sub_(rhs_update)  # [M*d, 1]

    try:
        U = torch.linalg.cholesky(scale_shift_hessian)
        dx = torch.cholesky_solve(scale_shift_rhs, U)  # [M*d, 1]
    except RuntimeError:
        dx = torch.zeros(
            M, d, device=scale_shift_rhs.device, dtype=scale_shift_rhs.dtype
        )
        dz = torch.zeros(M, hw, device=depth_rhs.device, dtype=depth_rhs.dtype)
        return dx, dz

    dz = inv_depth_diag * (depth_rhs - cross_termT @ dx)  # [M*hw, 1]
    dx = dx.reshape(M, d)
    dz = dz.reshape(M, hw)

    return dx, dz


def ba_ss(
    target: torch.Tensor,  # [E, ht, wd, 2] - target optical flow per edge
    weight: torch.Tensor,  # [E, ht, wd, 2] - confidence weights per edge
    damping: torch.Tensor,  # [T, ht, wd] - regularization per frame
    poses: Pose,  # [T] - camera poses (q: [T, 4], t: [T, 3])
    disps: torch.Tensor,  # [T, ht, wd] - disparity maps per frame
    intrinsics: Intrinsics,  # camera intrinsics (fx, fy, cx, cy)
    ii: torch.Tensor,  # [E] - source frame indices for each edge
    jj: torch.Tensor,  # [E] - destination frame indices for each edge
    mono_disps: torch.Tensor,  # [T, ht, wd] - monocular depth predictions
    scales: torch.Tensor,  # [T] - scale parameters for depth alignment
    shifts: torch.Tensor,  # [T] - shift parameters for depth alignment
    valid_depth_mask: torch.Tensor,  # [T, ht, wd] - valid depth mask per frame
    ignore_frames: int = 0,
    lm: float = 0.0001,  # Levenberg-Marquardt damping
    ep: float = 0.1,  # epsilon for numerical stability
    alpha: float = 1.0,  # weight for depth regularization
):
    """Bundle adjustment with scale and shift optimization.

    Optimize disparities, scales, and shifts together (eq.17 in the paper).
    """

    _, ht, wd = disps.shape

    # Get unique keyframes and create index mappings
    keyframe_indices, edge_to_keyframe = torch.unique(ii, return_inverse=True)
    damping_keyframes = 0.2 * damping[keyframe_indices].contiguous() + 1e-7

    # Prepare scale/shift parameters
    scale_shift_params = torch.stack([scales, shifts], dim=-1)  # [T, 2]

    # ========== PROJECTIVE JACOBIANS ==========
    proj_depth_diag, proj_depth_residual = projective_jacobians(
        poses, disps, intrinsics, ii, jj, target, weight
    )
    # Ck [E, hw]
    # wk [E, hw]

    # ========== DEPTH JACOBIANS ==========
    scale_shift_jac, depth_jac, depth_residual = depth_jacobians(
        disps,
        mono_disps,
        scales,
        shifts,
        valid_depth_mask,
        ignore_frames,
        alpha,
        indices=keyframe_indices,
    )
    # Jwq [M, hw, 2]
    # Jd [M, hw]
    # Rd [M, hw]

    # ========== LINEAR SYSTEM CONSTRUCTION ==========
    scale_shift_hessian, scale_shift_rhs, cross_term, depth_diag, depth_rhs = (
        assemble_scale_shift_sys(
            scale_shift_jac,
            depth_jac,
            depth_residual,
            proj_depth_diag,
            proj_depth_residual,
            damping_keyframes,
            edge_to_keyframe,
        )
    )
    # H [M, M, 2, 2]
    # u [M, 2]
    # E [M, M, 2, hw]
    # C [M, hw]
    # w [M, hw]

    # ========== SOLVE LINEAR SYSTEM ==========
    delta_scale_shift, delta_depth = schur_solve(
        scale_shift_hessian, cross_term, depth_diag, scale_shift_rhs, depth_rhs, ep, lm
    )
    # dwq [M, 2]
    # dz [M, hw]

    # ========== APPLY UPDATES ==========
    disps_out = disps.index_add(0, keyframe_indices, delta_depth.view(-1, ht, wd))
    disps_out.clamp_(min=0.0)

    scale_shift_params.index_add_(0, keyframe_indices, delta_scale_shift)

    return disps_out, scale_shift_params
