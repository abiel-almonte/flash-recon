"""
CUDA-accelerated projective operations
"""

from __future__ import annotations
import torch

__all__: list[str] = ["proj_cuda", "proj_jac_cuda"]

def proj_cuda(
    points: torch.Tensor, fx: float, fy: float, cx: float, cy: float, last_dim: int
) -> torch.Tensor:
    """
    Projection
    """

def proj_jac_cuda(
    points: torch.Tensor, fx: float, fy: float, cx: float, cy: float
) -> torch.Tensor:
    """
    Jacobian of projection
    """
