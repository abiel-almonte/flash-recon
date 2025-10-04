import torch

from pose_utils import Pose, Tangent, Intrinsics, pose_retraction
from projective_utils import projective_transform
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


def compute_motion_residuals_and_jacobians(
    poses: torch.Tensor,
    disps: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
):
    e = ii.shape[0]

    residuals, weights, Jj, Ji, _ = projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True
    )
    weights = 0.001 * (weights * weight)  # [E, H, W, 2]
    residuals.mul_(-1).add_(target)  # [E, H, W, 2]

    residuals = residuals.view(e, -1, 1)  # [E, 2*ht*wd, 1]
    weights = weights.view(e, -1, 1)  # [E, 2*ht*wd, 1]

    return residuals, weights, Jj, Ji


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


def assemble_motion_only_sys(
    jacobian_pose_src: torch.Tensor,  # [E, H, W, 2, 6]
    jacobian_pose_tgt: torch.Tensor,  # [E, H, W, 2, 6]
    weights: torch.Tensor,  # [E, 2*H*W, 1]
    residuals: torch.Tensor,  # [E, 2*H*W, 1]
    rig_size: int,
    num_fixed_poses: int,
    source_indices: torch.Tensor,  # [E]
    target_indices: torch.Tensor,  # [E]
    n_poses: int,
):

    n_edges = source_indices.size(0)
    manifold_dim = 6  # SE(3) dim

    num_opt_poses = n_poses // rig_size - num_fixed_poses
    src_opt_indices = source_indices // rig_size - num_fixed_poses
    tgt_opt_indices = target_indices // rig_size - num_fixed_poses

    jacobian_pose_src = jacobian_pose_src.reshape(
        n_edges, -1, manifold_dim
    )  # [E, 2*ht*wd, 6]
    jacobian_pose_tgt = jacobian_pose_tgt.reshape(
        n_edges, -1, manifold_dim
    )  # [E, 2*ht*wd, 6]
    weighted_jac_srcT = (weights * jacobian_pose_src).transpose(
        -2, -1
    )  # [E, 6, 2*ht*wd]
    weighted_jac_tgtT = (weights * jacobian_pose_tgt).transpose(
        -2, -1
    )  # [E, 6, 2*ht*wd]

    rhs = torch.cat([jacobian_pose_src, jacobian_pose_tgt], dim=-1)  # [E, 2*ht*wd, 12]
    lhs = torch.cat([weighted_jac_srcT, weighted_jac_tgtT], dim=-2)  # [E, 12, 2*ht*wd]

    hessian_12x12 = torch.bmm(lhs, rhs)  # [E, 12, 12]
    gradient_12 = torch.bmm(lhs, residuals).squeeze(-1)  # [E, 12]

    hessian = torch.zeros(
        (num_opt_poses * num_opt_poses, manifold_dim, manifold_dim),
        device=jacobian_pose_src.device,
        dtype=jacobian_pose_src.dtype,
    )

    gradient: torch.Tensor = torch.zeros(
        (num_opt_poses, manifold_dim),
        device=jacobian_pose_src.device,
        dtype=jacobian_pose_src.dtype,
    )

    src_valid = (src_opt_indices >= 0) & (src_opt_indices < num_opt_poses)
    tgt_valid = (tgt_opt_indices >= 0) & (tgt_opt_indices < num_opt_poses)

    valid_src_src = src_valid
    valid_src_tgt = src_valid & tgt_valid
    valid_tgt_tgt = tgt_valid

    hessian.index_add_(
        0,
        (
            src_opt_indices[valid_src_src] * num_opt_poses
            + src_opt_indices[valid_src_src]
        ),
        hessian_12x12[valid_src_src, :manifold_dim, :manifold_dim],
    )

    hessian.index_add_(
        0,
        (
            src_opt_indices[valid_src_tgt] * num_opt_poses
            + tgt_opt_indices[valid_src_tgt]
        ),
        hessian_12x12[valid_src_tgt, :manifold_dim, manifold_dim:],
    )

    hessian.index_add_(
        0,
        (
            tgt_opt_indices[valid_src_tgt] * num_opt_poses
            + src_opt_indices[valid_src_tgt]
        ),
        hessian_12x12[valid_src_tgt, manifold_dim:, :manifold_dim],
    )

    hessian.index_add_(
        0,
        (
            tgt_opt_indices[valid_tgt_tgt] * num_opt_poses
            + tgt_opt_indices[valid_tgt_tgt]
        ),
        hessian_12x12[valid_tgt_tgt, manifold_dim:, manifold_dim:],
    )

    gradient.index_add_(
        0, src_opt_indices[src_valid], gradient_12[src_valid, :manifold_dim]
    )
    gradient.index_add_(
        0, tgt_opt_indices[tgt_valid], gradient_12[tgt_valid, manifold_dim:]
    )

    hessian = hessian.view(num_opt_poses, num_opt_poses, manifold_dim, manifold_dim)

    return hessian, gradient


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


def block_solve(
    motion_only_hessian: torch.Tensor,
    gradient: torch.Tensor,
    ep: float = 0.1,
    lm: float = 0.0001,
):
    """solve normal equations"""

    num_opt_poses, _, manifold_dim, _ = motion_only_hessian.shape

    # Apply damping: H = H + (ep + lm*H)*I
    diag_view = motion_only_hessian.diagonal(dim1=-2, dim2=-1)  # [N, N, 6]
    diag_view.mul_(1 + lm).add_(ep)

    motion_only_hessian = motion_only_hessian.permute(0, 2, 1, 3).reshape(
        num_opt_poses * manifold_dim, num_opt_poses * manifold_dim
    )
    gradient = gradient.reshape(num_opt_poses * manifold_dim, 1)

    try:
        U = torch.linalg.cholesky(motion_only_hessian)
        pose_update = torch.cholesky_solve(gradient, U)  # [np*md, 1]
    except RuntimeError:
        pose_update = torch.zeros(
            num_opt_poses, manifold_dim, device=gradient.device, dtype=gradient.dtype
        )
        return pose_update

    pose_update = pose_update.reshape(num_opt_poses, manifold_dim)
    return pose_update


def ba_scale_shift(
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


def motion_only_ba(
    target: torch.Tensor,  # [E, H, W, 2] - target optical flow per edge
    weight: torch.Tensor,  # [E, H, W, 2] - confidence weights per edge
    poses: Pose,  # [T] - camera poses (q: [T, 4], t: [T, 3])
    disps: torch.Tensor,  # [T, H, W] - disparity maps per frame
    intrinsics: Intrinsics,  # camera intrinsics (fx, fy, cx, cy)
    source_indices: torch.Tensor,  # [E] - source frame indices for each edge
    target_indices: torch.Tensor,  # [E] - target frame indices for each edge
    num_fixed_poses: int = 1,  # number of fixed poses at the start
    rig_size: int = 1,  # rig size for multi-camera systems
):
    """Motion only bundle adjustment.
    
    Optimize camera poses while keeping disparities fixed.
    """
    n_poses = disps.size(0)
    manifold_dim = 6

    # ========== PROJECTIVE JACOBIANS ==========
    residuals, weights, jacobian_pose_src, jacobian_pose_tgt = (
        compute_motion_residuals_and_jacobians(
            poses, disps, intrinsics, source_indices, target_indices, target, weight
        )
    )
    # r [E, 2*H*W, 1]
    # w [E, 2*H*W, 1]
    # Ji [E, H, W, 2, 6]
    # Jj [E, H, W, 2, 6]

    # ========== LINEAR SYSTEM CONSTRUCTION ==========
    hessian, gradient = assemble_motion_only_sys(
        jacobian_pose_src,
        jacobian_pose_tgt,
        weights,
        residuals,
        rig_size,
        num_fixed_poses,
        source_indices,
        target_indices,
        n_poses,
    )
    # H [P, P, 6, 6]
    # v [P, 6]

    # ========== SOLVE LINEAR SYSTEM ==========
    pose_update = block_solve(hessian, gradient)
    # dx [P, 6]

    # ========== APPLY UPDATES ==========
    full_pose_update = torch.zeros(
        (n_poses, manifold_dim), device=disps.device, dtype=disps.dtype
    )
    full_pose_update[num_fixed_poses : num_fixed_poses + pose_update.size(0)] = (
        pose_update
    )

    tangent = Tangent(full_pose_update[:, :3], full_pose_update[:, 3:])
    updated_poses = pose_retraction(poses, tangent)

    return updated_poses
