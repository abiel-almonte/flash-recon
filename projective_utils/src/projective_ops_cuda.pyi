"""
CUDA-accelerated projective operations
"""

from __future__ import annotations
import torch

__all__: list[str] = [
    "fused_induced_flow_cuda",
    "fused_projective_cuda",
    "fused_projective_jac_cuda",
    "proj_cuda",
    "proj_jac_cuda",
]

def fused_induced_flow_cuda(
    t: torch.Tensor,
    q: torch.Tensor,
    disps: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Fused induced flow (iproj+SE3+proj)
    """

def fused_projective_cuda(
    t: torch.Tensor,
    q: torch.Tensor,
    disps: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Fused projective transform (iproj+SE3+proj)
    """

def fused_projective_jac_cuda(
    t: torch.Tensor,
    q: torch.Tensor,
    disps: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Fused projective transform with jacobians (iproj+SE3+proj)
    """

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
