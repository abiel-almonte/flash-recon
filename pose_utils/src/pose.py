import torch
from dataclasses import dataclass

@dataclass
class Pose:
    t: torch.Tensor  # [B, 3] translations
    q: torch.Tensor  # [B, 4] quaternions [qx, qy, qz, qw]. w is last!

    def __post_init__(self):
        self.q = self.q / torch.norm(self.q, dim=-1, keepdim=True)

    def __getitem__(self, idx):
        return Pose(self.t[idx], self.q[idx])

    @property
    def device(self):
        return self.t.device

    @property
    def dtype(self):
        return self.t.dtype
