import torch

from ..structs import Pose, Intrinsics

from geometry_cuda.ba import (
    fused_projective_transform_with_reduction_cuda,
    fused_depth_jacobians_cuda,
    fused_project_and_accumulate_cuda,
    scatter_pose_system_cuda,
)


def projective_transform_with_reduction(
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


def depth_jacobians(
    disps: torch.Tensor,
    mono_depths: torch.Tensor,
    scales: torch.Tensor,
    shifts: torch.Tensor,
    depth_mask: torch.Tensor,
    frame_bound: int,
    alpha: float,
    indices: torch.Tensor,
):
    disps_i = disps[indices].contiguous()
    mono_depths_i = mono_depths[indices].contiguous()
    depth_mask_i = depth_mask[indices].contiguous()
    scales_i = scales[indices].contiguous()
    shifts_i = shifts[indices].contiguous()
    ignore_i = indices < frame_bound

    return fused_depth_jacobians_cuda(
        disps_i, mono_depths_i, depth_mask_i, scales_i, shifts_i, ignore_i, alpha
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
    poses: Pose,
    disps: torch.Tensor,
    intrinsics: Intrinsics,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    damping: torch.Tensor,
    mono_depths: torch.Tensor,
    scales: torch.Tensor,
    shifts: torch.Tensor,
    vmask: torch.Tensor,
    ignore_frames: int,
    alpha: float,
    keyframe_indices: torch.Tensor,
    edge_to_keyframe: torch.Tensor,
):
    proj_depth_diag, proj_depth_residual = projective_transform_with_reduction(
        poses, disps, intrinsics, source_indices, target_indices, target, weight
    )
    # Ck [E, hw]
    # wk [E, hw]

    scale_shift_jac, depth_jac, depth_residual = depth_jacobians(
        disps,
        mono_depths,
        scales,
        shifts,
        vmask,
        ignore_frames,
        alpha,
        indices=keyframe_indices,
    )
    # Jwq [M, hw, 2]
    # Jd [M, hw]
    # Rd [M, hw]

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
    ep: float = 0.1,
    lm: float = 0.0001,
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
        ret_cross12=False,
    )
    (
        hessian,
        gradient,
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
        ep=ep,
        lm=lm,
    )
    return hessian, gradient


def assemble_full_sys(
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


def block_solve(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
):
    """Solve normal equations for block-structured system."""

    num_opt_poses, _, manifold_dim, _ = hessian.shape

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
