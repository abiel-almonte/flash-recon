from dataclasses import dataclass
from functools import cached_property
import torch


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    def __iter__(self):
        yield self.fx
        yield self.fy
        yield self.cx
        yield self.cy

    @cached_property
    def as_matrix(self):
        """Return the 3x3 camera intrinsics matrix."""
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=self.dtype,
            device=self.device,
        )

    @cached_property
    def as_tensor(self):
        """Return [fx, fy, cx, cy] tensor."""
        return torch.tensor(
            [self.fx, self.fy, self.cx, self.cy], dtype=self.dtype, device=self.device
        )

    @classmethod
    def from_matrix(cls, K: torch.Tensor):
        """Create from 3x3 intrinsics matrix."""
        return cls(
            fx=K[0, 0].item(),
            fy=K[1, 1].item(),
            cx=K[0, 2].item(),
            cy=K[1, 2].item(),
            device=str(K.device),
            dtype=K.dtype,
        )

    def to(self, device=None, dtype=None):
        """Return a new Intrinsics instance on a different device/dtype."""
        return Intrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            device=str(device or self.device),
            dtype=dtype or self.dtype,
        )

    def scale_resolution(self, scale_x: float, scale_y: float = None):
        """Scale intrinsics for image resolution changes."""
        if scale_y is None:
            scale_y = scale_x
        return Intrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            device=self.device,
            dtype=self.dtype,
        )

    def adjust_crop(self, dx: float, dy: float):
        """Adjust principal point for image cropping/padding."""
        return Intrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx + dx,
            cy=self.cy + dy,
            device=self.device,
            dtype=self.dtype,
        )

    def downsample(self, factor: float):
        """Downsample intrinsics."""
        return Intrinsics(
            fx=self.fx / factor,
            fy=self.fy / factor,
            cx=self.cx / factor,
            cy=self.cy / factor,
            device=self.device,
            dtype=self.dtype,
        )

    def to_fov(self, width: int, height: int):
        """Compute field of view angles in radians."""
        fovx = 2 * torch.atan(torch.tensor(width / (2 * self.fx), dtype=self.dtype))
        fovy = 2 * torch.atan(torch.tensor(height / (2 * self.fy), dtype=self.dtype))
        return fovx.item(), fovy.item()
