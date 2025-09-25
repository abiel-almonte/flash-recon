import torch

from pose_utils import Pose, Intrinsics
from ba_ops_cuda import (
    fused_projective_transform_with_reduction_cuda,
    fused_depth_jacobians_cuda,
    fused_scatter_ops_cuda,
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
    curv, rhs = fused_projective_transform_with_reduction_cuda(
        poses.t, poses.q, depths, intrinsics.as_tensor, ii, jj, target, weight
    )
    return curv, rhs


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

    scale_shift_hessian = torch.empty(
        (M, M, 2, 2), device=scale_shift_jac.device, dtype=scale_shift_jac.dtype
    )
    cross_term = torch.empty(
        (M, M, 2, hw), device=scale_shift_jac.device, dtype=scale_shift_jac.dtype
    )

    proj_residual_scatter = torch.zeros(
        (M, hw), device=scale_shift_jac.device, dtype=scale_shift_jac.dtype
    )
    proj_diag_scatter = torch.zeros(
        (M, hw), device=scale_shift_jac.device, dtype=scale_shift_jac.dtype
    )

    scale_shift_jac_depth_residual = torch.empty(
        (M, hw, 3), device=scale_shift_jac.device, dtype=scale_shift_jac.dtype
    )
    scale_shift_jac_depth_residual[..., :2] = scale_shift_jac
    scale_shift_jac_depth_residual[..., 2:3] = depth_residual.unsqueeze(-1)

    scale_shift_jacT = scale_shift_jac.transpose(-2, -1).contiguous()  # [M, 2, hw]
    prod = torch.bmm(scale_shift_jacT, scale_shift_jac_depth_residual)  # [M, 2, 3]

    scale_shift_hessian[idx, idx] = prod[..., :2]  # [M, M, 2, 2]
    scale_shift_rhs = -prod[..., 2:3].squeeze(-1)  # [M, 2]
    cross_term[idx, idx] = scale_shift_jacT * depth_jac.unsqueeze(1)  # [M, M, 2, hw]

    proj_residual_scatter.index_add_(0, kk, proj_depth_residual)  # [M, hw]
    proj_diag_scatter.index_add_(0, kk, proj_depth_diag)  # [M, hw]

    depth_diag = (
        proj_diag_scatter + depth_jac.pow(2) + eta.view(*proj_depth_diag.shape)
    )  # [M, hw]
    depth_rhs = -(proj_residual_scatter + depth_jac * depth_residual)  # [M, hw]

    return scale_shift_hessian, scale_shift_rhs, cross_term, depth_diag, depth_rhs


def bass(
    target: torch.Tensor,
    weight: torch.Tensor,
    eta,
    poses: Pose,
    disps: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
    mono_disps: torch.Tensor,
    scales: torch.Tensor,
    shifts: torch.Tensor,
    valid_depth_mask: torch.Tensor,
    ignore_frames: int,
    lm=0.0001,
    ep=0.1,
    alpha=1.0,
    fixedp=1,
    rig=1,
):
    """Bundle adjustment with scale and shift optimization.

    Optimize disparities, scales, and shifts together (eq.17 in the paper).
    Math details can be found in the supplementary material.
    """

    _, ht, wd = disps.shape

    # Normalize indices for rig/fixedp consistencyno
    ii = torch.div(ii, rig, rounding_mode="trunc") - fixedp
    jj = torch.div(jj, rig, rounding_mode="trunc") - fixedp

    # Get unique keyframes and create index mappings
    kx, kk = torch.unique(ii, return_inverse=True)

    # Prepare scale/shift parameters and monocular depth
    wqs = torch.stack([scales, shifts], dim=2)  # [P,2]

    # ========== PROJECTIVE JACOBIANS ==========
    proj_depth_diag, proj_depth_residual = projective_jacobians(
        poses, disps, intrinsics, ii, jj, target, weight
    )
    # Ck [P, hw]
    # wk [P, hw]

    # ========== DEPTH JACOBIANS ==========
    scale_shift_jac, depth_jac, depth_residual = depth_jacobians(
        disps,
        mono_disps,
        scales,
        shifts,
        valid_depth_mask,
        ignore_frames,
        alpha,
        indices=kx,
    )
    # Jwq [M, hw, 2]
    # Jd [M, hw]
    # Jr [M, hw]

    # ========== LINEAR SYSTEM CONSTRUCTION ==========
    scale_shift_hessian, scale_shift_rhs, cross_term, depth_diag, depth_rhs = (
        assemble_scale_shift_sys(
            scale_shift_jac,
            depth_jac,
            depth_residual,
            proj_depth_diag,
            proj_depth_residual,
            eta,
            kk,
        )
    )

    # ========== SOLVE LINEAR SYSTEM ==========
    dwq, dz = schur_solve(
        scale_shift_hessian, cross_term, depth_diag, scale_shift_rhs, depth_rhs, ep, lm
    )
    # dwq [B,M,2] - scale/shift updates
    # dz [B,M,hw] - depth updates

    # ========== APPLY UPDATES ==========
    disps.index_add_(0, kx, dz.view(-1, ht, wd))
    disps.clamp_(min=0.0)

    wqs.index_add_(0, kx, dwq)

    return poses, disps, wqs
