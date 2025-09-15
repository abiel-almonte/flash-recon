"""
CUDA-accelerated bundle adjustment operations
"""
from __future__ import annotations
import torch
__all__: list[str] = ['fused_projective_transform_with_reduction_cuda']
def fused_projective_transform_with_reduction_cuda(t: torch.Tensor, q: torch.Tensor, disps: torch.Tensor, intrinsics: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> list[torch.Tensor]:
    """
    Fused projective transform with reduction (iproj+SE3+proj)
    """
