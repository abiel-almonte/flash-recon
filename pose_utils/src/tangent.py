import torch
from dataclasses import dataclass


@dataclass
class Tangent:
    rho: torch.Tensor  # [B, 3]
    phi: torch.Tensor  # [B, 3]

    def __post_init__(self):
        if self.rho.ndim != 2 or self.rho.size(-1) != 3:
            raise ValueError(f"rho must be of shape [B, 3], got {self.rho.shape}")
        if self.phi.ndim != 2 or self.phi.size(-1) != 3:
            raise ValueError(f"phi must be of shape [B, 3], got {self.phi.shape}")
        if self.rho.shape[0] != self.phi.shape[0]:
            raise ValueError(
                f"rho and phi must have same batch size, got {self.rho.shape[0]} vs {self.phi.shape[0]}"
            )
        if self.rho.device != self.phi.device:
            raise ValueError(
                f"rho and phi must be on the same device, got {self.rho.device} vs {self.phi.device}"
            )
        if self.rho.dtype != self.phi.dtype:
            raise ValueError(
                f"rho and phi must have the same dtype, got {self.rho.dtype} vs {self.phi.dtype}"
            )

        self.rho = self.rho.contiguous()
        self.phi = self.phi.contiguous()

    def __getitem__(self, idx):
        return Tangent(self.rho[idx], self.phi[idx])

    @property
    def device(self):
        return self.rho.device

    @property
    def dtype(self):
        return self.rho.dtype

    def to(self, device: str):
        self.rho = self.rho.to(device)
        self.rhoh = self.phi.to(device)

    def __repr__(self):
        return f"Tangent(rho={self.rho}, phi={self.phi})"
