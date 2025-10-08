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


def compute_residuals_and_jacobians(
    poses: Pose,
    disps: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
):
    e = ii.shape[0]

    residuals, weights, Ji, Jj, Jz = projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True
    )
    weights = 0.001 * (weights * weight)  # [E, H, W, 2]
    residuals.mul_(-1).add_(target)  # [E, H, W, 2]

    residuals = residuals.view(e, -1, 1)  # [E, 2*ht*wd, 1]
    weights = weights.view(e, -1, 1)  # [E, 2*ht*wd, 1]

    return residuals, weights, Ji, Jj, Jz


def compute_edge_hessian_gradient(
    jacobian_pose_src: torch.Tensor,  # [E, H, W, 2, 6]
    jacobian_pose_tgt: torch.Tensor,  # [E, H, W, 2, 6]
    weights: torch.Tensor,  # [E, 2*H*W, 1]
    residuals: torch.Tensor,  # [E, 2*H*W, 1]
):
    """Compute per-edge 12x12 hessian and 12-dim gradient.

    Returns:
        hessian_12x12: [E, 12, 12]
        gradient_12: [E, 12]
        jacobian_pose_src_reshaped: [E, 2*ht*wd, 6]
        jacobian_pose_tgt_reshaped: [E, 2*ht*wd, 6]
    """
    n_edges = jacobian_pose_src.size(0)
    manifold_dim = 6

    jacobian_pose_src = jacobian_pose_src.reshape(n_edges, -1, manifold_dim)
    jacobian_pose_tgt = jacobian_pose_tgt.reshape(n_edges, -1, manifold_dim)

    weighted_jac_srcT = (weights * jacobian_pose_src).transpose(-2, -1)  # [E, 6, 2*ht*wd]
    weighted_jac_tgtT = (weights * jacobian_pose_tgt).transpose(-2, -1)  # [E, 6, 2*ht*wd]

    # Compute Hessian blocks separately for numerical stability (matches reference)
    Hii = torch.bmm(weighted_jac_srcT, jacobian_pose_src)  # [E, 6, 6]
    Hij = torch.bmm(weighted_jac_srcT, jacobian_pose_tgt)  # [E, 6, 6]
    Hji = torch.bmm(weighted_jac_tgtT, jacobian_pose_src)  # [E, 6, 6]
    Hjj = torch.bmm(weighted_jac_tgtT, jacobian_pose_tgt)  # [E, 6, 6]

    hessian_12x12 = torch.cat([
        torch.cat([Hii, Hij], dim=-1),  # [E, 6, 12]
        torch.cat([Hji, Hjj], dim=-1)   # [E, 6, 12]
    ], dim=-2)  # [E, 12, 12]

    # Gradient
    vi = torch.bmm(weighted_jac_srcT, residuals).squeeze(-1)  # [E, 6]
    vj = torch.bmm(weighted_jac_tgtT, residuals).squeeze(-1)  # [E, 6]
    gradient_12 = torch.cat([vi, vj], dim=-1)  # [E, 12]

    return hessian_12x12, gradient_12, jacobian_pose_src, jacobian_pose_tgt


def scatter_pose_system(
    hessian_12x12: torch.Tensor,  # [E, 12, 12]
    gradient_12: torch.Tensor,  # [E, 12]
    source_indices: torch.Tensor,  # [E]
    target_indices: torch.Tensor,  # [E]
    num_opt_poses: int,
    rig_size: int,
    num_fixed_poses: int,
):
    """Scatter per-edge hessian/gradient into global pose system.

    Returns:
        hessian: [P, P, 6, 6]
        gradient: [P, 6]
        src_opt_indices: [E]
        tgt_opt_indices: [E]
        src_valid: [E]
        tgt_valid: [E]
    """
    manifold_dim = 6

    src_opt_indices = source_indices // rig_size - num_fixed_poses
    tgt_opt_indices = target_indices // rig_size - num_fixed_poses

    src_valid = (src_opt_indices >= 0) & (src_opt_indices < num_opt_poses)
    tgt_valid = (tgt_opt_indices >= 0) & (tgt_opt_indices < num_opt_poses)

    valid_src_src = src_valid
    valid_src_tgt = src_valid & tgt_valid
    valid_tgt_tgt = tgt_valid

    hessian = torch.zeros(
        (num_opt_poses * num_opt_poses, manifold_dim, manifold_dim),
        device=hessian_12x12.device,
        dtype=hessian_12x12.dtype,
    )

    hessian.index_add_(
        0,
        src_opt_indices[valid_src_src] * num_opt_poses + src_opt_indices[valid_src_src],
        hessian_12x12[valid_src_src, :manifold_dim, :manifold_dim],
    )

    hessian.index_add_(
        0,
        src_opt_indices[valid_src_tgt] * num_opt_poses + tgt_opt_indices[valid_src_tgt],
        hessian_12x12[valid_src_tgt, :manifold_dim, manifold_dim:],
    )

    hessian.index_add_(
        0,
        tgt_opt_indices[valid_src_tgt] * num_opt_poses + src_opt_indices[valid_src_tgt],
        hessian_12x12[valid_src_tgt, manifold_dim:, :manifold_dim],
    )

    hessian.index_add_(
        0,
        tgt_opt_indices[valid_tgt_tgt] * num_opt_poses + tgt_opt_indices[valid_tgt_tgt],
        hessian_12x12[valid_tgt_tgt, manifold_dim:, manifold_dim:],
    )

    hessian = hessian.view(num_opt_poses, num_opt_poses, manifold_dim, manifold_dim)

    gradient = torch.zeros(
        (num_opt_poses, manifold_dim),
        device=hessian_12x12.device,
        dtype=hessian_12x12.dtype,
    )

    gradient.index_add_(
        0, src_opt_indices[src_valid], gradient_12[src_valid, :manifold_dim]
    )
    gradient.index_add_(
        0, tgt_opt_indices[tgt_valid], gradient_12[tgt_valid, manifold_dim:]
    )

    return hessian, gradient, src_opt_indices, tgt_opt_indices, src_valid, tgt_valid

def compute_depth_terms(
    jacobian_pose_src: torch.Tensor,  # [E, 2*ht*wd, 6]
    jacobian_pose_tgt: torch.Tensor,  # [E, 2*ht*wd, 6]
    jacobian_depth: torch.Tensor,  # [E, H, W, 2, 1]
    weights: torch.Tensor,  # [E, 2*H*W, 1]
    residuals: torch.Tensor,  # [E, 2*H*W, 1]
    ht: int,
    wd: int,
):
    """Compute depth-related terms for full BA.

    Returns:
        cross_term_12: [E, 12, ht*wd] - pose-depth cross terms
        depth_diag: [E, ht*wd] - depth hessian diagonal
        depth_residual: [E, ht*wd] - depth gradient
    """
    n_edges = jacobian_depth.size(0)
    manifold_dim = 6

    jacobian_depth = jacobian_depth.reshape(n_edges, ht * wd, -1)  # [E, ht*wd, 2]

    # Compute weighted Jacobian transposes (like reference)
    wJiT = (weights * jacobian_pose_src).transpose(1, 2)  # [E, 6, 2*ht*wd]
    wJjT = (weights * jacobian_pose_tgt).transpose(1, 2)  # [E, 6, 2*ht*wd]
    
    # Reshape for cross-term computation
    lhs = torch.cat([wJiT, wJjT], dim=1)  # [E, 12, 2*ht*wd]
    lhs = lhs.view(n_edges, 2 * manifold_dim, ht * wd, 2)  # [E, 12, ht*wd, 2]

    weights = weights.view(n_edges, ht * wd, 2)  # [E, ht*wd, 2]
    residuals = residuals.view(n_edges, ht * wd, 2)  # [E, ht*wd, 2]

    weighted_jacobian_depth = weights * jacobian_depth

    cross_term_12 = torch.sum(lhs * jacobian_depth.unsqueeze(1), dim=-1)  # [E, 12, ht*wd]
    depth_diag = torch.sum(
        jacobian_depth * weighted_jacobian_depth, dim=-1
    )  # [E, ht*wd]
    depth_residual = torch.sum(
        residuals * weighted_jacobian_depth, dim=-1
    )  # [E, ht*wd]
    return cross_term_12, depth_diag, depth_residual


def assemble_scale_shift_sys(
    scale_shift_jac: torch.Tensor,
    depth_jac: torch.Tensor,
    depth_residual: torch.Tensor,
    proj_depth_diag: torch.Tensor,
    proj_depth_residual: torch.Tensor,
    damping: torch.Tensor,
    edge_to_keyframe: torch.Tensor,
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

    hessian_update, gradient_update = prod.split([2, 1], dim=-1)
    scale_shift_hessian[idx, idx] = hessian_update  # [M, M, 2, 2]
    scale_shift_gradient = -gradient_update.squeeze(-1)  # [M, 2]
    cross_term[idx, idx] = scale_shift_jacT * depth_jac.unsqueeze(1)  # [M, M, 2, hw]

    proj_residual_scatter.index_add_(
        0, edge_to_keyframe, proj_depth_residual
    )  # [M, hw]
    proj_diag_scatter.index_add_(0, edge_to_keyframe, proj_depth_diag)  # [M, hw]

    depth_diag = (
        proj_diag_scatter + depth_jac.pow(2) + damping.view(*proj_depth_diag.shape)
    )  # [M, hw]
    depth_gradient = -(proj_residual_scatter + depth_jac * depth_residual)  # [M, hw]

    return (
        scale_shift_hessian,
        scale_shift_gradient,
        cross_term,
        depth_diag,
        depth_gradient,
    )


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
    num_opt_poses = n_poses // rig_size - num_fixed_poses

    hessian_12x12, gradient_12, _, _ = compute_edge_hessian_gradient(
        jacobian_pose_src, jacobian_pose_tgt, weights, residuals
    )

    hessian, gradient, _, _, _, _ = scatter_pose_system(
        hessian_12x12,
        gradient_12,
        source_indices,
        target_indices,
        num_opt_poses,
        rig_size,
        num_fixed_poses,
    )

    return hessian, gradient

def assemble_full_ba_sys(
    jacobian_pose_src: torch.Tensor,  # [E, H, W, 2, 6]
    jacobian_pose_tgt: torch.Tensor,  # [E, H, W, 2, 6]
    jacobian_depth: torch.Tensor,  # [E, H, W, 2, 1]
    weights: torch.Tensor,  # [E, 2*H*W, 1]
    residuals: torch.Tensor,  # [E, 2*H*W, 1]
    source_indices: torch.Tensor,  # [E]
    target_indices: torch.Tensor,  # [E]
    edge_to_keyframe: torch.Tensor,  # [E]
    keyframe_indices: torch.Tensor,  # [M]
    damping: torch.Tensor,  # [T, ht, wd]
    rig_size: int,
    num_fixed_poses: int,
    n_poses: int,
    ht: int,
    wd: int,
):
    """Assemble linear system for full BA (pose + depth).

    Returns:
        hessian: [P, P, 6, 6]
        gradient: [P, 6]
        cross_term: [P, M, 6, ht*wd]
        depth_diag: [M, ht*wd]
        depth_gradient: [M, ht*wd]
    """
    manifold_dim = 6
    num_opt_poses = n_poses // rig_size - num_fixed_poses
    M = keyframe_indices.shape[0]

    hessian_12x12, gradient_12, jacobian_pose_src, jacobian_pose_tgt = (
        compute_edge_hessian_gradient(
            jacobian_pose_src, jacobian_pose_tgt, weights, residuals
        )
    )

    hessian, gradient, src_opt_indices, tgt_opt_indices, src_valid, tgt_valid = (
        scatter_pose_system(
            hessian_12x12,
            gradient_12,
            source_indices,
            target_indices,
            num_opt_poses,
            rig_size,
            num_fixed_poses,
        )
    )

    cross_term_12, depth_diag, depth_residual = compute_depth_terms(
        jacobian_pose_src, jacobian_pose_tgt, jacobian_depth, weights, residuals, ht, wd
    )


    cross_term = torch.zeros(
        (num_opt_poses * M, manifold_dim, ht * wd),
        device=jacobian_pose_src.device,
        dtype=jacobian_pose_src.dtype,
    )

    cross_term.index_add_(
        0,
        src_opt_indices[src_valid] * M + edge_to_keyframe[src_valid],
        cross_term_12[src_valid, :manifold_dim],
    )

    cross_term.index_add_(
        0,
        tgt_opt_indices[tgt_valid] * M + edge_to_keyframe[tgt_valid],
        cross_term_12[tgt_valid, manifold_dim:],
    )

    cross_term = cross_term.view(num_opt_poses, M, manifold_dim, ht * wd)

    # Accumulate depth diagonal and gradient
    depth_diag_accum = torch.zeros_like(depth_diag[:M])
    depth_gradient_accum = torch.zeros_like(depth_residual[:M])

    depth_diag_accum.index_add_(0, edge_to_keyframe, depth_diag)
    depth_diag_accum.add_(damping[keyframe_indices].view(M, ht * wd))

    depth_gradient_accum.index_add_(0, edge_to_keyframe, depth_residual)

    return hessian, gradient, cross_term, depth_diag_accum, depth_gradient_accum


def schur_solve(
    hessian: torch.Tensor,
    cross_term: torch.Tensor,
    depth_diag: torch.Tensor,
    gradient: torch.Tensor,
    depth_gradient: torch.Tensor,
    ep: float = 0.1,
    lm: float = 0.0001,
):
    """Solve linear system using Schur complement.

    Generic solver that works for both scale/shift BA and full BA.
    
    Args:
        hessian: [P, P, d, d] pose Hessian
        cross_term: [P, M, d, hw] pose-depth cross term
        depth_diag: [M, hw] depth diagonal
        gradient: [P, d] pose gradient
        depth_gradient: [M, hw] depth gradient
    """

    P, M, d, hw = cross_term.shape
    hessian = hessian.permute(0, 2, 1, 3).reshape(P * d, P * d)  # [P*d, P*d]
    cross_term = cross_term.permute(0, 2, 1, 3).reshape(P * d, M * hw)  # [P*d, M*hw]
    inv_depth_diag = (1.0 / depth_diag).view(M * hw, 1)
    gradient = gradient.reshape(P * d, 1)
    depth_gradient = depth_gradient.reshape(M * hw, 1)

    # Apply damping - match reference: H = H + (ep + lm*H)*I
    I = torch.eye(P * d, device=hessian.device, dtype=hessian.dtype)
    hessian = hessian + (ep + lm * hessian) * I

    cross_termT = cross_term.transpose(-2, -1)  # [M*hw, P*d]
    schur_hessian = hessian - torch.matmul(cross_term, inv_depth_diag * cross_termT)
    schur_gradient = gradient - torch.matmul(cross_term, inv_depth_diag * depth_gradient)

    try:
        U = torch.linalg.cholesky(schur_hessian)
        dx = torch.cholesky_solve(schur_gradient, U)  # [P*d, 1]
    except RuntimeError:
        dx = torch.zeros(P, d, device=gradient.device, dtype=gradient.dtype)
        dz = torch.zeros(
            M, hw, device=depth_gradient.device, dtype=depth_gradient.dtype
        )
        return dx, dz

    dz = inv_depth_diag * (depth_gradient - cross_termT @ dx)  # [M*hw, 1]
    dx = dx.reshape(P, d)
    dz = dz.reshape(M, hw)

    return dx, dz


def block_solve(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    ep: float = 0.1,
    lm: float = 0.0001,
):
    """Solve normal equations for block-structured system."""

    num_opt_poses, _, manifold_dim, _ = hessian.shape

    # Reshape before damping to match reference structure
    hessian = hessian.permute(0, 2, 1, 3).reshape(
        num_opt_poses * manifold_dim, num_opt_poses * manifold_dim
    )
    gradient = gradient.reshape(num_opt_poses * manifold_dim, 1)

    # Apply damping - match reference: H = H + (ep + lm*H)*I  
    I = torch.eye(num_opt_poses * manifold_dim, device=hessian.device, dtype=hessian.dtype)
    hessian = hessian + (ep + lm * hessian) * I

    try:
        U = torch.linalg.cholesky(hessian)
        update = torch.cholesky_solve(gradient, U)  # [np*md, 1]
    except RuntimeError:
        update = torch.zeros(
            num_opt_poses, manifold_dim, device=gradient.device, dtype=gradient.dtype
        )
        return update

    update = update.reshape(num_opt_poses, manifold_dim)
    return update


def ba_scale_shift(
    target: torch.Tensor,  # [E, ht, wd, 2] - target optical flow per edge
    weight: torch.Tensor,  # [E, ht, wd, 2] - confidence weights per edge
    damping: torch.Tensor,  # [T, ht, wd] - regularization per frame
    poses: Pose,  # [T] - camera poses (q: [T, 4], t: [T, 3])
    disps: torch.Tensor,  # [T, ht, wd] - disparity maps per frame
    intrinsics: Intrinsics,  # camera intrinsics (fx, fy, cx, cy)
    source_indices: torch.Tensor,  # [E] - source frame indices for each edge
    target_indices: torch.Tensor,  # [E] - target frame indices for each edge
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
    keyframe_indices, edge_to_keyframe = torch.unique(
        source_indices, return_inverse=True
    )
    damping_keyframes = 0.2 * damping[keyframe_indices].contiguous() + 1e-7

    # Prepare scale/shift parameters
    scale_shift_params = torch.stack([scales, shifts], dim=-1)  # [T, 2]

    # ========== PROJECTIVE JACOBIANS ==========
    proj_depth_diag, proj_depth_residual = projective_transform_with_reduction_fused(
        poses, disps, intrinsics, source_indices, target_indices, target, weight
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
    (
        scale_shift_hessian,
        scale_shift_gradient,
        cross_term,
        depth_diag,
        depth_gradient,
    ) = assemble_scale_shift_sys(
        scale_shift_jac,
        depth_jac,
        depth_residual,
        proj_depth_diag,
        proj_depth_residual,
        damping_keyframes,
        edge_to_keyframe,
    )
    # scale_shift_hessian [M, M, 2, 2]
    # scale_shift_gradient [M, 2]
    # cross_term [M, M, 2, hw]
    # depth_diag [M, hw]
    # depth_gradient [M, hw]

    # ========== SOLVE LINEAR SYSTEM ==========
    delta_scale_shift, delta_depth = schur_solve(
        scale_shift_hessian,
        cross_term,
        depth_diag,
        scale_shift_gradient,
        depth_gradient,
        ep,
        lm,
    )
    # delta_scale_shift [M, 2]
    # delta_depth [M, hw]

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
    residuals, weights, jacobian_pose_src, jacobian_pose_tgt, _ = (
        compute_residuals_and_jacobians(
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
    # hessian [P, P, 6, 6]
    # gradient [P, 6]

    # ========== SOLVE LINEAR SYSTEM ==========
    update = block_solve(hessian, gradient)
    # update [P, 6]

    # ========== APPLY UPDATES ==========
    full_update = torch.zeros(
        (n_poses, manifold_dim), device=disps.device, dtype=disps.dtype
    )
    full_update[num_fixed_poses : num_fixed_poses + update.size(0)] = update

    tangent = Tangent(full_update[:, :3], full_update[:, 3:])
    updated_poses = pose_retraction(poses, tangent)

    return updated_poses

def full_ba(
    target: torch.Tensor,  # [E, H, W, 2] - target optical flow per edge
    weight: torch.Tensor,  # [E, H, W, 2] - confidence weights per edge
    damping: torch.Tensor,  # [T, H, W] - regularization per frame
    poses: Pose,  # [T] - camera poses (q: [T, 4], t: [T, 3])
    disps: torch.Tensor,  # [T, H, W] - disparity maps per frame
    intrinsics: Intrinsics,  # camera intrinsics (fx, fy, cx, cy)
    source_indices: torch.Tensor,  # [E] - source frame indices for each edge
    target_indices: torch.Tensor,  # [E] - target frame indices for each edge
    lm: float = 0.0001,  # Levenberg-Marquardt damping
    ep: float = 0.1,  # epsilon for numerical stability
    alpha: float = 0.05,  # weight for depth regularization (unused in current impl)
    num_fixed_poses: int = 1,  # number of fixed poses at the start
    rig_size: int = 1,  # rig size for multi-camera systems
):
    """Full bundle adjustment.

    Jointly optimize camera poses and disparities.
    """

    n_poses, ht, wd = disps.shape
    manifold_dim = 6

    keyframe_indices, edge_to_keyframe = torch.unique(
        source_indices, return_inverse=True
    )

    # ========== PROJECTIVE JACOBIANS ==========
    residuals, weights, jacobian_pose_src, jacobian_pose_tgt, jacobian_depth = (
        compute_residuals_and_jacobians(
            poses, disps, intrinsics, source_indices, target_indices, target, weight
        )
    )
    # r [E, 2*H*W, 1]
    # w [E, 2*H*W, 1]
    # Ji [E, H, W, 2, 6]
    # Jj [E, H, W, 2, 6]
    # Jz [E, H, W, 2, 1]

    # ========== LINEAR SYSTEM CONSTRUCTION ==========
    hessian, gradient, cross_term, depth_diag, depth_gradient = assemble_full_ba_sys(
        jacobian_pose_src,
        jacobian_pose_tgt,
        jacobian_depth,
        weights,
        residuals,
        source_indices,
        target_indices,
        edge_to_keyframe,
        keyframe_indices,
        damping,
        rig_size,
        num_fixed_poses,
        n_poses,
        ht,
        wd,
    )
    # hessian [P, P, 6, 6]
    # gradient [P, 6]
    # cross_term [P, M, 6, ht*wd]
    # depth_diag [M, ht*wd]
    # depth_gradient [M, ht*wd]

    # ========== SOLVE LINEAR SYSTEM ==========
    pose_update, depth_update = schur_solve(
        hessian, cross_term, depth_diag, gradient, depth_gradient, ep, lm
    )
    # pose_update [P, 6]
    # depth_update [M, ht*wd]

    # ========== APPLY UPDATES ==========

    full_update = torch.zeros(
        (n_poses, manifold_dim), device=disps.device, dtype=disps.dtype
    )
    full_update[num_fixed_poses : num_fixed_poses + pose_update.size(0)] = pose_update

    tangent = Tangent(full_update[:, :3], full_update[:, 3:])
    poses_out = pose_retraction(poses, tangent)

    disps_out = disps.index_add(0, keyframe_indices, depth_update.view(-1, ht, wd))
    disps_out.clamp_(min=0.0)

    return poses_out, disps_out

"""
Hii, Hij, Hjj, vi, vj,     \
cross12_or_none, ddiag, dres = \
    fused_project_and_accumulate_cuda(poses.t, poses.q,
                                      disps, intrinsics.as_tensor,
                                      ii, jj, target, weight,
                                      output_cross_term=True)

H = fused_scatter_pose_blocks_cuda(
      Hii, Hij, Hjj, vi, vj,
      src_idx, tgt_idx, P)  # returns H[P,P,6,6], g[P,6]

cross_PM = fused_scatter_cross_terms_cuda(
    cross12, src_opt_idx, tgt_opt_idx, edge_to_keyframe, P, M)

Hss, gss, cross_sHw, D, gD = fused_assemble_scale_shift_cuda(
    Jwq, Jd, Rd, Ck, wk, damping_keyframes, edge_to_keyframe)

y = fused_schur_matvec_cuda(x, H_blocks, JiJjHandles, DinvHandles, grouping)
"""