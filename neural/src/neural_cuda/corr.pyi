from __future__ import annotations
import torch

__all__: list[str] = ["altcorr_forward", "corr_forward"]

def altcorr_forward(
    fmap1: torch.Tensor, fmap2: torch.Tensor, coords: torch.Tensor, radius: int
) -> torch.Tensor:
    """
    Compute correlations on the fly
    """

def corr_forward(
    volume: torch.Tensor, coords: torch.Tensor, radius: int
) -> torch.Tensor:
    """
    Compute correlations
    """
