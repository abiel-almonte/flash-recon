from dataclasses import dataclass
from typing import Optional
from .roles import CallerRole


@dataclass
class OptimizationPayload:
    """Payload containing all data needed for bundle adjustment optimization."""

    # Caller role
    role: CallerRole

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
    mono_disps: Optional[object] = None
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
