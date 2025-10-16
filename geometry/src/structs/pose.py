import torch
from dataclasses import dataclass


@dataclass
class Pose:
    t: torch.Tensor  # [B, 3] translations
    q: torch.Tensor  # [B, 4] quaternions [qx, qy, qz, qw]. w is last!

    def __post_init__(self):
        if self.t.ndim != 2 or self.t.size(-1) != 3:
            raise ValueError(f"t must be of shape [B, 3], got {self.t.shape}")
        if self.q.ndim != 2 or self.q.size(-1) != 4:
            raise ValueError(f"q must be of shape [B, 4], got {self.q.shape}")
        if self.t.shape[0] != self.q.shape[0]:
            raise ValueError(
                f"t and q must have same batch size, got {self.t.shape[0]} vs {self.q.shape[0]}"
            )
        if self.t.device != self.q.device:
            raise ValueError(
                f"t and q must be on the same device, got {self.t.device} vs {self.q.device}"
            )
        if self.t.dtype != self.q.dtype:
            raise ValueError(
                f"t and q must have the same dtype, got {self.t.dtype} vs {self.q.dtype}"
            )

        self.q = self.q / torch.norm(self.q, dim=-1, keepdim=True)

        self.q = self.q.contiguous()
        self.t = self.t.contiguous()

    def __getitem__(self, idx):
        return Pose(self.t[idx], self.q[idx])

    def __setitem__(self, idx, pose: "Pose"):
        if isinstance(pose, "Pose"):
            self.t[idx] = pose.t
            self.q[idx] = pose.q

        else:
            raise ValueError(f"pose must be of type Pose. Got {pose.__class__}")

    @property
    def device(self):
        return self.t.device

    @property
    def dtype(self):
        return self.t.dtype

    def to(self, device: str):
        self.t = self.t.to(device)
        self.q = self.q.to(device)

    def __repr__(self):
        return f"Pose(t={self.t}, q={self.q})"
