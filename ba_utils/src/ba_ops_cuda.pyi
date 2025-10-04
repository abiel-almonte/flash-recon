"""
CUDA-accelerated bundle adjustment operations
"""

from __future__ import annotations
import torch

__all__: list[str] = [
    "fused_depth_jacobians_cuda",
    "fused_projective_transform_with_reduction_cuda",
]

def fused_depth_jacobians_cuda(
    disps: torch.Tensor,
    mono_disps: torch.Tensor,
    valid_depth: torch.Tensor,
    scales: torch.Tensor,
    shifts: torch.Tensor,
    ignore: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused operation to compute depth jacobians
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
    Fused projective transform with reduction (iproj+SE3+proj)
    """
