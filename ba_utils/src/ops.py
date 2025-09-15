import torch

from pose_utils import Pose, Intrinsics
from ba_ops_cuda import fused_projective_transform_with_reduction_cuda


def projective_transform_with_reduction_fused(
    poses: Pose,
    depths: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor
):
    curv, rhs = fused_projective_transform_with_reduction_cuda(
        poses.t, poses.q, depths, intrinsics.as_tensor, ii, jj, target, weight
    )
    return curv, rhs


def projective_transform_with_reduction(
    poses: Pose,
    depths: torch.Tensor,
    intrinsics: Intrinsics,
    ii: torch.Tensor,
    jj: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor
):
    return projective_transform_with_reduction_fused(poses, depths, intrinsics, ii, jj, target, weight)


def ba_ss(
    target : torch.Tensor,
    weight : torch.Tesnor,
    eta,
    poses : Pose,
    disps : torch.Tensor,
    intrinsics : Intrinsics,
    ii : torch.Tensor,
    jj : torch.Tensor,
    mono_disps : torch.Tensor,
    scales : torch.Tensor,
    shifts : torch.Tensor,
    valid_depth_mask : torch.Tensor,
    ignore_frames : int,
    lm=0.0001,
    ep=0.1,
    alpha=1.0,
    fixedp=1,
    rig=1,
):
    """optimize disparities (disp), scales (w) and shifts (q) together, eq.17 in the paper,
    math details can be found in the supplementary
    """

    device = ii.device
    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim
    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]
    sqrt_alpha = torch.tensor(alpha).sqrt().to(device)
    ll = torch.arange(M, device=device)
    wqs = torch.stack([scales, shifts], dim=2)  # [B,P,2]

    ignore_mask = kx < ignore_frames
    invalid_mask = (mono_disps[:, kx] < 1e-6).view(B, M, ht * wd)  # [B,M,ht*wd]
    invalid_mask[:, ignore_mask] = True

    valid_depth_mask = valid_depth_mask[:, kx].view(B, M, ht * wd)
    Ck, wk = projective_transform_with_reduction(poses, disps, intrinsics, ii, jj, target, weight)

    r_depth = sqrt_alpha * (
        disps[:, kx]
        - (scales[:, kx, None, None] * mono_disps[:, kx] + shifts[:, kx, None, None])
    ).view(B, M, ht * wd, 1)

    sqrt_alpha = torch.ones(B, M, ht * wd, 1).float().to(device) * sqrt_alpha
    sqrt_alpha[valid_depth_mask] *= 10

    J_d = torch.ones(B, M, ht * wd, 1).float().to(device) * sqrt_alpha
    J_scale = (
        -mono_disps[:, kx].clone().view(B, M, ht * wd, 1) * sqrt_alpha
    )  # [B,M,ht*wd,1]
    J_shift = (
        -torch.ones(B, M, ht * wd, 1).float().to(device) * sqrt_alpha
    )  # [B,M,ht*wd,1]

    J_d[invalid_mask * valid_depth_mask] = 0
    J_scale[invalid_mask] = 0
    J_shift[invalid_mask] = 0

    J_wq = torch.cat([J_scale, J_shift], dim=3)  # [B,M,ht*wd,2]
    J_wq_T = J_wq.transpose(2, 3)  # [B,M,2,ht*wd]
    H_wq = torch.matmul(J_wq_T, J_wq)  # [B,M,2,2]
    u = -torch.matmul(J_wq_T, r_depth).squeeze(-1)  # [B,M,2]
    ### 2: construct linear system ###

    Jz = Jz.reshape(B, N, ht * wd, -1)  # [B,N,ht*wd,2]
    # here Jz does not contain the negative sign in the residual term

    E_wq_d = (J_wq_T.view(B, M, 2, ht * wd, -1) * J_d[:, :, None]).sum(
        dim=-1
    )  # [B,M,2,ht*wd]

    # only optimize keyframe poses
    P = torch.div(P, rig, rounding_mode="trunc") - fixedp
    ii = torch.div(ii, rig, rounding_mode="trunc") - fixedp
    jj = torch.div(jj, rig, rounding_mode="trunc") - fixedp

    H_wq = safe_scatter_add_mat(H_wq, ll, ll, M, M)  # [B,M*M,2,2]
    E_wq_d = safe_scatter_add_mat(E_wq_d, ll, ll, M, M)  # [B,M*M,2,ht*wd]
    C_proj = safe_scatter_add_vec(Ck, kk, M)  # [B,M,ht*wd]
    u = safe_scatter_add_vec(u, ll, M)  # [B,M,2]

    # C = C + eta.view(*C.shape) #+ 1e-7
    C_depth = (J_d * J_d).view(B, M, ht * wd)
    # C = C_proj + C_depth + (1-C_depth)*eta.view(*C_proj.shape)             #[B,M,ht*wd]
    C = C_proj + C_depth + eta.view(*C_proj.shape)  # + 1e-7

    w_proj = safe_scatter_add_vec(wk, kk, M)  # [B,M,ht*wd]
    w = -w_proj - (J_d * r_depth).view(B, M, ht * wd)  # [B,M,ht*wd]
    H = H_wq.view(B, M, M, 2, 2)
    E = E_wq_d.view(B, M, M, 2, ht * wd)
    ### 3: solve the system ###
    dwq, dz = schur_solve(H, E, C, u, w, ep, lm)
    # dwq [B,M,2]
    # dz [B,M,ht*wd]
    ### 4: apply retraction ###
    # poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B, -1, ht, wd), kx)
    wqs = wq_retr(wqs, dwq, kx)
    # disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    return poses, disps, wqs
