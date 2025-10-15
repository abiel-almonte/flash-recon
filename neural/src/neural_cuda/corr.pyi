from __future__ import annotations
import torch

__all__: list[str] = ["corr_forward"]

def corr_forward(
    volume: torch.Tensor, coords: torch.Tensor, radius: int
) -> torch.Tensor:
    """
    Compute correlations
    """
