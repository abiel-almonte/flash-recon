from enum import Enum


class BAType(Enum):
    POSE_DEPTH = "pose_depth"
    DEPTH_SCALE = "depth_scale"
    MOTION_ONLY = "motion_only"
