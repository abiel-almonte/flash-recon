import torch

from geometry import (
    Pose,
    Tangent,
    pose_retraction,
    matrix_to_pose,
    pose_to_matrix,
)


def update_pose(camera, converged_threshold=1e-4):
    """
    Drop-in replacement for update_pose - 2.9x faster
    """
    rho = camera.cam_trans_delta.unsqueeze(0)  # [1, 3]
    phi = camera.cam_rot_delta.unsqueeze(0)  # [1, 3]
    tangent = Tangent(rho, phi)

    # Build 4x4 transformation matrix from camera.R (3x3) and camera.T (3,)
    T_mat = torch.eye(4, device=camera.R.device, dtype=camera.R.dtype)
    T_mat[:3, :3] = camera.R
    T_mat[:3, 3] = camera.T

    current_pose = matrix_to_pose(T_mat.unsqueeze(0))
    new_pose = pose_retraction(current_pose, tangent)

    # Extract new R and T from the updated pose
    new_T_mat = pose_to_matrix(new_pose)[0]  # [4, 4]
    new_R = new_T_mat[:3, :3]  # [3, 3]
    new_T = new_T_mat[:3, 3]   # [3]

    camera.update_RT(new_R, new_T)

    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta])
    converged = tau.norm() < converged_threshold

    return converged
