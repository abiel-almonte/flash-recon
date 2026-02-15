import torch

from typing import Optional
from dataclasses import dataclass


@dataclass
class SplatSnapshot:
    n: Optional[int] = None
    means: Optional[torch.Tensor] = None
    colors: Optional[torch.Tensor] = None
    quats: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None
    alphas: Optional[torch.Tensor] = None

    def detach(self):
        return SplatSnapshot(
            n=self.n,
            means=self.means.detach(),
            colors=self.colors.detach(),
            quats=self.quats.detach(),
            scales=self.scales.detach(),
            alphas=self.alphas.detach(),
        )
