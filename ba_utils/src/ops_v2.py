import torch

torch.set_float32_matmul_precision("high")

from pose_utils import Pose, Tangent, Intrinsics, pose_retraction
from ba_ops_cuda import (
    fused_projective_transform_with_reduction_cuda,
    fused_depth_jacobians_cuda,
    fused_project_and_accumulate_cuda,
    scatter_pose_system_cuda,
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


def fused_project_and_accumulate(
    poses: Pose,
    disps: torch.Tensor,
    intr: Intrinsics,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    ret_cross12: torch.Tensor,
):

    return fused_project_and_accumulate_cuda(
        poses.t,
        poses.q,
        disps,
        intr.as_tensor,
        source_indices,
        target_indices,
        target,
        weight,
        ret_cross12,
    )


def scatter_pose_system(
    Hii: torch.Tensor,  # [E, 6, 6]
    Hij: torch.Tensor,  # [E, 6, 6]
    Hji: torch.Tensor,  # [E, 6, 6]
    Hjj: torch.Tensor,  # [E, 6, 6]
    vi: torch.Tensor,  # [E, 6]
    vj: torch.Tensor,  # [E, 6]
    source_indices: torch.Tensor,  # [E]
    target_indices: torch.Tensor,  # [E]
    num_opt_poses: int,
    rig_size: int,
    num_fixed_poses: int,
    Ei: torch.Tensor = None,  # [E, 6, hw]
    Ej: torch.Tensor = None,  # [E, 6, hw]
    Ck: torch.Tensor = None,  # [E, hw]
    wk: torch.Tensor = None,  # [E, hw]
    ret_cross: bool = False,
    ret_depth: bool = False,
    edge_to_keyframe: torch.Tensor = None,  # [E]
    keyframe_indices: torch.Tensor = None,  # [M]
    damping: torch.Tensor = None,  # [T, ht, wd]
    M: int = 1,
    ht: int = 1,
    wd: int = 1,
    ep: float = 0.0,
    lm: float = 0.0,
):
    """Scatter per-edge hessian/gradient (and optionally cross-terms and depth terms) into global BA system.
    Applies damping to Hessian if ep > 0 or lm > 0.

    Returns:
        if ret_cross=False, ret_depth=False:
            hessian: [P, P, 6, 6]
            gradient: [P, 6]

        if ret_cross=True, ret_depth=False:
            hessian: [P, P, 6, 6]
            gradient: [P, 6]
            cross_term: [P, M, 6, hw]

        if ret_cross=False, ret_depth=True:
            hessian: [P, P, 6, 6]
            gradient: [P, 6]
            depth_diag: [M, hw]
            depth_gradient: [M, hw]

        if ret_cross=True, ret_depth=True:
            hessian: [P, P, 6, 6]
            gradient: [P, 6]
            cross_term: [P, M, 6, hw]
            depth_diag: [M, hw]
            depth_gradient: [M, hw]
    """

    E = Hii.shape[0]

    if Ei is None or not ret_cross:
        Ei = torch.empty((E, 6, 1), device=Hii.device, dtype=Hii.dtype)
        Ej = torch.empty((E, 6, 1), device=Hii.device, dtype=Hii.dtype)

    if Ck is None or not ret_depth:
        Ck = torch.empty((E, 1), device=Hii.device, dtype=Hii.dtype)
        wk = torch.empty((E, 1), device=Hii.device, dtype=Hii.dtype)

    if edge_to_keyframe is None:
        edge_to_keyframe = torch.empty((E,), device=Hii.device, dtype=torch.long)

    if keyframe_indices is None:
        keyframe_indices = torch.empty((1,), device=Hii.device, dtype=torch.long)

    if damping is None:
        damping = torch.empty((1, 1, 1), device=Hii.device, dtype=Hii.dtype)

    results = scatter_pose_system_cuda(
        Hii,
        Hij,
        Hji,
        Hjj,
        vi,
        vj,
        Ei,
        Ej,
        Ck,
        wk,
        source_indices,
        target_indices,
        edge_to_keyframe,
        keyframe_indices,
        damping,
        num_opt_poses,
        rig_size,
        num_fixed_poses,
        ret_cross,
        ret_depth,
        M,
        ht,
        wd,
        ep,
        lm,
    )

    if ret_cross and ret_depth:
        hessian, gradient, cross_term, depth_diag, depth_gradient = results
        return hessian, gradient, cross_term, depth_diag, depth_gradient
    elif ret_cross:
        hessian, gradient, cross_term = results
        return hessian, gradient, cross_term
    elif ret_depth:
        hessian, gradient, depth_diag, depth_gradient = results
        return hessian, gradient, depth_diag, depth_gradient
    else:
        hessian, gradient = results
        return hessian, gradient


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
    device = scale_shift_jac.device
    dtype = scale_shift_jac.dtype

    proj_residual_scatter = torch.zeros((M, hw), device=device, dtype=dtype)
    proj_diag_scatter = torch.zeros((M, hw), device=device, dtype=dtype)
    proj_residual_scatter.index_add_(0, edge_to_keyframe, proj_depth_residual)
    proj_diag_scatter.index_add_(0, edge_to_keyframe, proj_depth_diag)

    J_reshaped = scale_shift_jac.permute(1, 0, 2).reshape(hw, M * 2)  # [hw, M*2]
    hessian_flat = J_reshaped.T @ J_reshaped  # [M*2, M*2]

    hessian_reshaped = hessian_flat.reshape(M, 2, M, 2)
    idx = torch.arange(M, device=device)
    scale_shift_hessian_diag = hessian_reshaped[idx, :, idx, :]  # [M, 2, 2]

    scale_shift_jacT = scale_shift_jac.transpose(-2, -1)  # [M, 2, hw]
    scale_shift_gradient = -torch.bmm(
        scale_shift_jacT, depth_residual.unsqueeze(-1)
    ).squeeze(
        -1
    )  # [M, 2]

    cross_term_diag = scale_shift_jacT * depth_jac.unsqueeze(1)  # [M, 2, hw]

    depth_diag = proj_diag_scatter + depth_jac.pow(2) + damping.view(M, hw)
    depth_gradient = -(proj_residual_scatter + depth_jac * depth_residual)

    scale_shift_hessian = torch.zeros((M, M, 2, 2), device=device, dtype=dtype)
    cross_term = torch.zeros((M, M, 2, hw), device=device, dtype=dtype)
    scale_shift_hessian[idx, idx] = scale_shift_hessian_diag
    cross_term[idx, idx] = cross_term_diag

    return (
        scale_shift_hessian,
        scale_shift_gradient,
        cross_term,
        depth_diag,
        depth_gradient,
    )


def assemble_motion_only_sys(
    poses: Pose,
    disps: torch.Tensor,
    intrinsics: Intrinsics,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    num_opt_poses: int,
    rig_size: int,
    num_fixed_poses: int,
):
    """Assemble linear system for motion-only BA."""
    Hii, Hij, Hji, Hjj, vi, vj, _, _ = fused_project_and_accumulate(
        poses,
        disps,
        intrinsics,
        source_indices,
        target_indices,
        target,
        weight,
        ret_cross12=False,  # Don't need cross terms for motion-only
    )
    (
        hessian,
        gradient,
        _,
    ) = scatter_pose_system(
        Hii,
        Hij,
        Hji,
        Hjj,
        vi,
        vj,
        source_indices,
        target_indices,
        num_opt_poses,
        rig_size,
        num_fixed_poses,
    )
    return hessian, gradient


def assemble_full_ba_sys(
    poses: Pose,
    disps: torch.Tensor,
    intr: Intrinsics,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    edge_to_keyframe: torch.Tensor,  # [E]
    keyframe_indices: torch.Tensor,  # [M]
    damping: torch.Tensor,  # [M, ht, wd] - damping for M source keyframes
    rig_size: int,
    num_fixed_poses: int,
    n_poses: int,
    ht: int,
    wd: int,
    ep: float = 0.1,
    lm: float = 0.0001,
):
    """Assemble linear system for full BA (pose + depth).

    Hessian damping (ep + lm*H)*I is applied.

    Returns:
        hessian: [P, P, 6, 6]  (with damping already applied)
        gradient: [P, 6]
        cross_term: [P, M, 6, ht*wd]
        depth_diag: [M, ht*wd]
        depth_gradient: [M, ht*wd]
    """

    num_opt_poses = n_poses // rig_size - num_fixed_poses
    M = keyframe_indices.shape[0]

    Hii, Hij, Hji, Hjj, vi, vj, Ei, Ej, Ck, wk = fused_project_and_accumulate(
        poses,
        disps,
        intr,
        source_indices,
        target_indices,
        target,
        weight,
        ret_cross12=True,
    )

    hessian, gradient, cross_term, depth_diag, depth_gradient = scatter_pose_system(
        Hii,
        Hij,
        Hji,
        Hjj,
        vi,
        vj,
        source_indices,
        target_indices,
        num_opt_poses,
        rig_size,
        num_fixed_poses,
        Ei=Ei,
        Ej=Ej,
        Ck=Ck,
        wk=wk,
        ret_cross=True,
        ret_depth=True,
        edge_to_keyframe=edge_to_keyframe,
        keyframe_indices=keyframe_indices,
        damping=damping,
        M=M,
        ht=ht,
        wd=wd,
        ep=ep,
        lm=lm,
    )

    return hessian, gradient, cross_term, depth_diag, depth_gradient


def _schur_solve_impl(
    hessian: torch.Tensor,
    cross_term: torch.Tensor,
    depth_diag: torch.Tensor,
    gradient: torch.Tensor,
    depth_gradient: torch.Tensor,
    ep: float = 0.1,
    lm: float = 0.0001,
):
    """Schur complement solver implementation."""
    P, M, d, hw = cross_term.shape
    hessian = hessian.permute(0, 2, 1, 3).reshape(P * d, P * d)  # [P*d, P*d]
    cross_term = cross_term.permute(0, 2, 1, 3).reshape(P * d, M * hw)  # [P*d, M*hw]
    inv_depth_diag = (1.0 / depth_diag).view(M * hw, 1)
    gradient = gradient.reshape(P * d, 1)
    depth_gradient = depth_gradient.reshape(M * hw, 1)

    if ep > 0 or lm > 0:
        diag_hess = torch.diagonal(hessian)
        damping_hess = ep + lm * diag_hess
        hessian = hessian + torch.diag(damping_hess)

    cross_termT = cross_term.transpose(-2, -1)  # [M*hw, P*d]
    schur_hessian = hessian - torch.matmul(cross_term, inv_depth_diag * cross_termT)
    schur_gradient = gradient - torch.matmul(
        cross_term, inv_depth_diag * depth_gradient
    )

    try:
        U = torch.linalg.cholesky(schur_hessian)
        dx = torch.cholesky_solve(schur_gradient, U)  # [P*d, 1]
    except RuntimeError:
        dx = None
        if d <= 2:
            diag_S = torch.diagonal(schur_hessian)
            scale = torch.clamp(diag_S.mean(), min=1e-6)
            I = torch.eye(P * d, device=schur_hessian.device, dtype=schur_hessian.dtype)
            for tau in (1e-6, 1e-4, 1e-2, 1e-1, 1.0):
                try:
                    S_damped = schur_hessian + (tau * scale) * I
                    U = torch.linalg.cholesky(S_damped)
                    dx = torch.cholesky_solve(schur_gradient, U)
                    break
                except RuntimeError:
                    continue
        if dx is None:
            dx = torch.zeros(P, d, device=gradient.device, dtype=gradient.dtype)
            dz = torch.zeros(
                M, hw, device=depth_gradient.device, dtype=depth_gradient.dtype
            )
            return dx, dz

    dz = inv_depth_diag * (depth_gradient - cross_termT @ dx)  # [M*hw, 1]
    dx = dx.reshape(P, d)
    dz = dz.reshape(M, hw)

    return dx, dz


@torch.compile(mode="reduce-overhead")
def schur_solve_compiled(
    hessian: torch.Tensor,
    cross_term: torch.Tensor,
    depth_diag: torch.Tensor,
    gradient: torch.Tensor,
    depth_gradient: torch.Tensor,
    ep: float = 0.1,
    lm: float = 0.0001,
):
    return _schur_solve_impl(
        hessian, cross_term, depth_diag, gradient, depth_gradient, ep, lm
    )


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
    Non-compiled version (used within scale_shift_assemble_and_solve).

    Args:
        hessian: [P, P, d, d] pose Hessian
        cross_term: [P, M, d, hw] pose-depth cross term
        depth_diag: [M, hw] depth diagonal
        gradient: [P, d] pose gradient
        depth_gradient: [M, hw] depth gradient
    """
    return _schur_solve_impl(
        hessian, cross_term, depth_diag, gradient, depth_gradient, ep, lm
    )


def block_solve(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
):
    """Solve normal equations for block-structured system.

    Note: If damping is already applied in the Hessian
    """

    num_opt_poses, _, manifold_dim, _ = hessian.shape

    # Reshape before damping to match reference structure
    hessian = hessian.permute(0, 2, 1, 3).reshape(
        num_opt_poses * manifold_dim, num_opt_poses * manifold_dim
    )
    gradient = gradient.reshape(num_opt_poses * manifold_dim, 1)

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


def scale_shift_assemble_and_solve(
    scale_shift_jac: torch.Tensor,
    depth_jac: torch.Tensor,
    depth_residual: torch.Tensor,
    proj_depth_diag: torch.Tensor,
    proj_depth_residual: torch.Tensor,
    damping: torch.Tensor,
    edge_to_keyframe: torch.Tensor,
    ep: float,
    lm: float,
):
    """Assemble scale/shift system and solve (compiled as single graph)."""
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
        damping,
        edge_to_keyframe,
    )

    delta_scale_shift, delta_depth = schur_solve(
        scale_shift_hessian,
        cross_term,
        depth_diag,
        scale_shift_gradient,
        depth_gradient,
        ep,
        lm,
    )

    return delta_scale_shift, delta_depth


def ba_scale_shift(
    target: torch.Tensor,  # [E, ht, wd, 2] - target optical flow per edge
    weight: torch.Tensor,  # [E, ht, wd, 2] - confidence weights per edge
    eta: torch.Tensor,  # [M, ht, wd] - damping for M source keyframes
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
    damping_keyframes = eta

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

    # ========== LINEAR SYSTEM CONSTRUCTION + SOLVE (COMPILED) ==========
    delta_scale_shift, delta_depth = scale_shift_assemble_and_solve(
        scale_shift_jac,
        depth_jac,
        depth_residual,
        proj_depth_diag,
        proj_depth_residual,
        damping_keyframes,
        edge_to_keyframe,
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
    num_opt_poses = n_poses // rig_size - num_fixed_poses

    # ========== FUSED PROJECTIVE JACOBIANS & HESSIAN ==========
    Hii, Hij, Hji, Hjj, vi, vj, _, _ = fused_project_and_accumulate(
        poses,
        disps,
        intrinsics,
        source_indices,
        target_indices,
        target,
        weight,
        ret_cross12=False,  # Don't need cross terms for motion-only
    )

    # ========== SCATTER TO GLOBAL SYSTEM ==========
    hessian, gradient = scatter_pose_system(
        Hii,
        Hij,
        Hji,
        Hjj,
        vi,
        vj,
        source_indices,
        target_indices,
        num_opt_poses,
        rig_size,
        num_fixed_poses,
        ret_cross=False,
        ret_depth=False,
        ep=0.1,
        lm=0.0001,
    )
    # hessian [P, P, 6, 6] (with damping already applied)
    # gradient [P, 6]

    # ========== SOLVE LINEAR SYSTEM ==========
    update = block_solve(hessian, gradient)  # Damping already applied

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
    eta: torch.Tensor,  # [M, H, W] - damping for M source keyframes
    poses: Pose,  # [T] - camera poses (q: [T, 4], t: [T, 3])
    disps: torch.Tensor,  # [T, H, W] - disparity maps per frame
    intrinsics: Intrinsics,  # camera intrinsics (fx, fy, cx, cy)
    source_indices: torch.Tensor,  # [E] - source frame indices for each edge
    target_indices: torch.Tensor,  # [E] - target frame indices for each edge
    n: int = 0,
    lm: float = 0.0001,  # Levenberg-Marquardt damping
    ep: float = 0.1,  # epsilon for numerical stability
    alpha: float = 0.05,  # weight for depth regularization (unused in current impl)
    num_fixed_poses: int = 1,  # number of fixed poses at the start
    rig_size: int = 1,  # rig size for multi-camera systems
):
    """Full bundle adjustment.

    Jointly optimize camera poses and disparities.
    """
    manifold_dim = 6
    T = poses.t.shape[0]

    if n == 0:
        n = T

    use_window = n < T

    if use_window:
        poses_window = poses[:n]
        disps_window = disps[:n]

        valid = (source_indices < n) & (target_indices < n)
        ii_window = source_indices[valid].contiguous()
        jj_window = target_indices[valid].contiguous()
        target_window = target[valid].contiguous()
        weight_window = weight[valid].contiguous()
    else:
        poses_window = poses
        disps_window = disps
        ii_window = source_indices
        jj_window = target_indices
        target_window = target
        weight_window = weight

    n_poses, ht, wd = disps_window.shape
    keyframe_indices, edge_to_keyframe = torch.unique(ii_window, return_inverse=True)

    damping_keyframes = 0.2 * eta + 1e-7

    # ========== PROJECTIVE JACOBIANS & LINEAR SYSTEM CONSTRUCTION ==========

    hessian, gradient, cross_term, depth_diag, depth_gradient = assemble_full_ba_sys(
        poses_window,
        disps_window,
        intrinsics,
        ii_window,
        jj_window,
        target_window,
        weight_window,
        edge_to_keyframe,
        keyframe_indices,
        damping_keyframes,
        rig_size,
        num_fixed_poses,
        n_poses,
        ht,
        wd,
        ep=0,  # Don't apply damping in CUDA - let schur_solve handle it
        lm=0,
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
    window_update = torch.zeros(
        (n_poses, manifold_dim), device=disps.device, dtype=disps.dtype
    )
    window_update[num_fixed_poses : num_fixed_poses + pose_update.size(0)] = pose_update

    tangent = Tangent(window_update[:, :3], window_update[:, 3:])
    poses_window_out = pose_retraction(poses_window, tangent)

    disps_window_out = disps_window.index_add(
        0, keyframe_indices, depth_update.view(-1, ht, wd)
    )
    disps_window_out.clamp_(min=0.0)

    if use_window:
        poses.t[:n] = poses_window_out.t
        poses.q[:n] = poses_window_out.q
        disps[:n] = disps_window_out
        return poses, disps
    else:
        return poses_window_out, disps_window_out
