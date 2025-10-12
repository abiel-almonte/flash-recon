import torch

from structs import Pose, Tangent, Intrinsics
from utils import (
    assemble_scale_shift_sys,
    assemble_full_sys,
    assemble_motion_only_sys,
    block_solve,
    schur_solve,
)

from .lie import pose_retraction


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

    keyframe_indices, edge_to_keyframe = torch.unique(
        source_indices, return_inverse=True
    )
    damping_keyframes = eta
    scale_shift_params = torch.stack([scales, shifts], dim=-1)  # [T, 2]

    (
        scale_shift_hessian,
        scale_shift_gradient,
        cross_term,
        depth_diag,
        depth_gradient,
    ) = assemble_scale_shift_sys(
        poses,
        disps,
        intrinsics,
        source_indices,
        target_indices,
        target,
        weight,
        damping_keyframes,
        mono_disps,
        scales,
        shifts,
        valid_depth_mask,
        ignore_frames,
        alpha,
        keyframe_indices,
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
    lm: float = 0.0001,
    ep: float = 0.1,
):
    """Motion only bundle adjustment.

    Optimize camera poses while keeping disparities fixed.
    """
    n_poses = disps.size(0)
    manifold_dim = 6
    num_opt_poses = n_poses // rig_size - num_fixed_poses

    hessian, gradient = assemble_motion_only_sys(
        poses,
        disps,
        intrinsics,
        source_indices,
        target_indices,
        target,
        weight,
        num_opt_poses,
        rig_size,
        num_fixed_poses,
        ep=ep,
        lm=lm,
    )
    # hessian [P, P, 6, 6]
    # gradient [P, 6]

    update = block_solve(hessian, gradient)

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

    hessian, gradient, cross_term, depth_diag, depth_gradient = assemble_full_sys(
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
