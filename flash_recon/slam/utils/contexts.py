import torch

from geometry import Pose, Intrinsics

from dataclasses import dataclass
from typing import Optional
from .enums import BAType


@dataclass
class BAContext:
    # ba type
    type: BAType

    # Core optimization data
    target: Optional[object] = None
    weight: Optional[object] = None
    eta: Optional[object] = None
    poses: Optional[object] = None
    disps: Optional[object] = None
    intrinsics: Optional[object] = None
    ii: Optional[object] = None
    jj: Optional[object] = None

    # Scale-shift specific data
    mono_depths: Optional[object] = None
    scales: Optional[object] = None
    shifts: Optional[object] = None
    valid_depth_mask: Optional[object] = None
    invalid_mono_frames: Optional[object] = None

    # Optimization parameters
    iters: int = 1
    ignore_frames: int = 1
    num_fixed_poses: int = 1
    rig_size: int = 1
    n: int = 0
    motion_only: bool = False

    # Solver parameters
    lm: Optional[float] = None
    ep: Optional[float] = None
    alpha: Optional[float] = None


@dataclass
class BufferSnapshot:
    count: Optional[int] = None

    poses: Optional[object] = None
    disps: Optional[object] = None
    intrinsics: Optional[object] = None

    fmaps: Optional[object] = None
    nets: Optional[object] = None
    inps: Optional[object] = None


@dataclass
class SLAMSnapshot:
    poses: Optional[Pose] = None
    disps: Optional[torch.Tensor] = None
    vmask: Optional[torch.Tensor] = None
    intrinsics: Optional[Intrinsics] = None
    version: Optional[object] = None

    @property
    def n_keyframes(self):
        return self.disps.size(0)

    def copy(self):
        return SLAMSnapshot(
            poses=self.poses.clone(),
            disps=self.disps.clone(),
            intrinsics=self.intrinsics,
            version=self.version,
        )


@dataclass
class KeyFrame:
    frame: Optional[object] = None
    poses: Optional[object] = None
    disps: Optional[object] = None  # disps up
    vmask: Optional[object] = None  # vmask up

    def add_frame(self, frame) -> None:
        self.frame = frame
