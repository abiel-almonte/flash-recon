"""
CUDA-accelerated Lie group operations for SO(3) and SE(3)
"""
from __future__ import annotations
import torch
__all__: list[str] = ['matrix_to_quat_cuda', 'quat_multiply_cuda', 'quat_rotate_cuda', 'quat_to_matrix_cuda', 'se3_exp_cuda', 'se3_log_cuda', 'se3_point_jac_cuda']
def matrix_to_quat_cuda(arg0: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion
    """
def quat_multiply_cuda(arg0: torch.Tensor, arg1: torch.Tensor) -> torch.Tensor:
    """
    Quaternion multiplication
    """
def quat_rotate_cuda(arg0: torch.Tensor, arg1: torch.Tensor) -> torch.Tensor:
    """
    Rotate points by quaternion
    """
def quat_to_matrix_cuda(arg0: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix
    """
def se3_exp_cuda(arg0: torch.Tensor, arg1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SE3 exponential map
    """
def se3_log_cuda(arg0: torch.Tensor, arg1: torch.Tensor) -> torch.Tensor:
    """
    SE3 logarithm map
    """
def se3_point_jac_cuda(arg0: torch.Tensor, arg1: int) -> torch.Tensor:
    """
    SE3 point jacobian
    """
