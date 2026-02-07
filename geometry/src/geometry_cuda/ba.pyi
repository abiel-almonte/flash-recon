"""
CUDA-accelerated bundle adjustment operations
"""

from __future__ import annotations
import torch

__all__: list[str] = [
    "fused_depth_jacobians_cuda",
    "fused_project_and_accumulate_cuda",
    "fused_projective_transform_with_reduction_cuda",
    "scatter_pose_system_cuda",
]

def fused_depth_jacobians_cuda(
    disps: torch.Tensor,
    mono_depths: torch.Tensor,
    valid_depth: torch.Tensor,
    scales: torch.Tensor,
    shifts: torch.Tensor,
    ignore: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused operation to compute depth jacobians
    """

def fused_project_and_accumulate_cuda(
    t: torch.Tensor,
    q: torch.Tensor,
    disps: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    ret_cross12: bool,
) -> list[torch.Tensor]:
    """
    Projective transform and assemble linear system
    """

def fused_projective_transform_with_reduction_cuda(
    t: torch.Tensor,
    q: torch.Tensor,
    disps: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused projective transform with reduction
    """

def scatter_pose_system_cuda(
    Hii: torch.Tensor,
    Hij: torch.Tensor,
    Hji: torch.Tensor,
    Hjj: torch.Tensor,
    vi: torch.Tensor,
    vj: torch.Tensor,
    Ei: torch.Tensor,
    Ej: torch.Tensor,
    Ck: torch.Tensor,
    wk: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    edge_to_keyframe: torch.Tensor,
    keyframe_indices: torch.Tensor,
    damping: torch.Tensor,
    num_opt_poses: int,
    rig_size: int,
    num_fixed_poses: int,
    ret_cross: bool,
    ret_depth: bool,
    M: int,
    ht: int,
    wd: int,
    ep: float,
    lm: float,
) -> list[torch.Tensor]:
    """
    Fused scatter of per-edge Hessians/gradients, cross-terms, and depth terms into global BA system with optional damping
    """
