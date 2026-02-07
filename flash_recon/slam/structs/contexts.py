from dataclasses import dataclass
from typing import Optional
from .enums import EdgeStrategy, BAType


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

    # Optimization parameters
    type: str = "poses_depths"
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
class EdgeRequest:

    strategy: EdgeStrategy

    buffer: Optional[BufferSnapshot] = None
    beta: float = 0.3
    t0_loop: Optional[int] = None

    t0: int = 0
    t1: int = 0
    rad: int = 2
    nms: int = 2
    thresh: float = 16.0
    max_factors: int = -1
    remove: bool = False
    loop: bool = False
