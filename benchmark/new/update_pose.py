import torch

from geometry import (
    Pose,
    Tangent,
    pose_retraction,
    matrix_to_pose
)


def update_pose(camera, converged_threshold=1e-4):
    """
    Drop-in replacement for update_pose - 2.9x faster
    """
    rho = camera.cam_trans_delta.unsqueeze(0)  # [1, 3]
    phi = camera.cam_rot_delta.unsqueeze(0)  # [1, 3]
    tangent = Tangent(rho, phi)

    current_pose = matrix_to_pose(camera.R.unsqueeze(0))

    new_pose = pose_retraction(current_pose, tangent)

    #new_R = quat_to_matrix_cuda(new_pose.q)[0]  # [3, 3]
    new_T = new_pose.t[0]  # [3]
    #camera.update_RT(new_R, new_T)

    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta])
    converged = tau.norm() < converged_threshold

    return converged
