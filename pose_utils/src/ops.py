import torch

from pose import Pose
from constants import (
    get_identity_pose,
    get_sign,
    get_eye4,
)

from lie_ops_cuda import (
    se3_log_cuda,
    se3_exp_cuda,
    se3_point_jac_cuda,
    quat_rotate_cuda, 
    quat_multiply_cuda, 
    quat_to_matrix_cuda,
    matrix_to_quat_cuda,
)

# Pure functional lie group operations

    
def identity_pose(batch_size: int, device='cuda', dtype= torch.float) -> Pose:
    """Create identity poses"""

    pose = get_identity_pose(device, dtype)
    pose.t= pose.t.repeat(batch_size, 1)
    pose.q =pose.q.repeat(batch_size, 1)

    return pose
  
def pose_inv(pose: Pose) -> Pose:
    """Invert SE3 poses.
    
    Implements SE3 inverse: SE3^-1 = SE3(R^T, -R^T * t)
    
    Mathematical derivation:
    - SE3 inverse: SE3(R, t)^-1 = SE3(R^T, -R^T * t)
    - Quaternion inverse: q^-1 = q* (conjugate) for unit quaternions
    - Quaternion conjugate: [qx, qy, qz, qw] -> [-qx, -qy, -qz, qw]
    - Rotated translation: R^T * (-t) using quaternion rotation
    
    Args:
        pose: Input SE3 pose to invert
        
    Returns:
        Inverted SE3 pose
    """

    sign = get_sign(pose.device, pose.dtype)
    q_inv = pose.q * sign
    t_inv = quat_rotate_cuda(q_inv, -pose.t)
    
    return Pose(t_inv, q_inv)

def pose_mul(pose1: Pose, pose2: Pose) -> Pose:
    """Compose SE3 poses: pose1 * pose2.
    
    Implements SE3 group multiplication: SE3(R1, t1) * SE3(R2, t2) = SE3(R1*R2, t1 + R1*t2)
    
    Mathematical derivation:
    - Rotation composition: R1 * R2 (quaternion multiplication)
    - Translation composition: t1 + R1 * t2 (rotation then translation)
    - Quaternion multiplication: Hamilton product
    
    Args:
        pose1: First SE3 pose (applied second)
        pose2: Second SE3 pose (applied first)
        
    Returns:
        Composed SE3 pose
    """

    t_new = pose1.t + quat_rotate_cuda(pose1.q, pose2.t)
    q_new = quat_multiply_cuda(pose1.q, pose2.q)

    return Pose(t_new, q_new)

def transform_points_by_pose(pose: Pose, points: torch.Tensor) -> torch.Tensor:
    """Transform 3D or 4D points by SE3 pose.
    
    Supports both Euclidean (3D) and homogeneous (4D) coordinates:
    - 3D points: p' = R * p + t
    - 4D points: [x', y', z', w'] = [R * [x,y,z] + t * w, w]
    
    Args:
        pose: SE3 pose containing rotation (quaternion) and translation
        points: Input points tensor [..., 3] or [..., 4]
        
    Returns:
        Transformed points with same shape as input
        
    Raises:
        AssertionError: If points are not 3D or 4D
    """

    if points.shape[-1] == 3:
        return quat_rotate_cuda(pose.q, points) + pose.t.unsqueeze(-2)
    
    elif points.shape[-1] == 4:
        out = torch.zeros_like(points)
        xyz = points[..., :3]
        w = points[..., 3:4]
        
        out[..., :3] = quat_rotate_cuda(pose.q, xyz) + pose.t.unsqueeze(-2) * w
        out[..., 3:4] = w
        return out
    
    else:
        raise ValueError(f"Points must be 3D or 4D, got shape {points.shape}")

def pose_to_tangent(pose: Pose) -> torch.Tensor:
    """SE3 logarithm map: SE3 -> se3 tangent space.
    
    Maps SE3 group elements to 6D tangent vectors [Rho, Phi] where:
    - Rho ∈ R3: translation component  
    - Phi ∈ R3: rotation component (axis-angle)
    
    Args:
        pose: SE3 pose to convert to tangent space
        
    Returns:
        6D tangent vector [Rhox, Rhoy, Rhoz, Phix, Phiy, Phiz]
    """
    
    return se3_log_cuda(pose.t, pose.q)

def tangent_to_pose(tangent: torch.Tensor) -> Pose:
    """SE3 exponential map: se3 tangent space -> SE3.
    
    Maps 6D tangent vectors [Rho, Phi] to SE3 group elements where:
    - Rho ∈ R3: translation component
    - Phi ∈ R3: rotation component (axis-angle)
    
    Args:
        tangent:    6D tangent vector [Rhox, Rhoy, Rhoz, Phix, Phiy, Phiz]
        
    Returns:
        SE3 pose corresponding to the tangent vector
    """

    rho = tangent[..., :3]
    phi = tangent[..., 3:]
    t, q = se3_exp_cuda(rho, phi)

    return Pose(t, q)

def pose_to_matrix(pose: Pose) -> torch.Tensor:
    """Convert SE3 pose to 4x4 transformation matrix.
    
    Converts SE3(R, t) to homogeneous transformation matrix:
    [[R, t],
     [0, 1]]
     
    Args:
        pose: SE3 pose to convert
        
    Returns:
        4x4 transformation matrix [B, 4, 4]
    """

    batch_size = pose.t.shape[0]
    rot_mat = quat_to_matrix_cuda(pose.q)  # [B, 3, 3]
    
    eye = get_eye4(pose.device, pose.t.type)
    T = eye.repeat(batch_size, 1, 1)
    T[..., :3, :3] = rot_mat
    T[..., :3, 3] = pose.t
    
    return T

def matrix_to_pose(T: torch.Tensor) -> Pose:
    """Convert 4x4 transformation matrix to SE3 pose.
    
    Converts homogeneous transformation matrix to SE3(R, t):
    [[R, t],
     [0, 1]]  -> [tx, ty, tz, qx, qy, qz, qw]
     
    Args:
        T: 4x4 transformation matrix to convert 
        
    Returns:
       SE3 pose
    """

    rot_matrix = T[..., :3, :3]
    t = T[..., :3, 3]
    q = matrix_to_quat_cuda(rot_matrix)

    return Pose(t, q)


def points_to_pose_jacobian(points : torch.Tensor) -> torch.Tensor:
    """Build SE3 action Jacobian for points.
    
    Args:
        points: [..., 4] tensor of homogeneous points
    Returns:
        [..., 4, 6] Jacobian wrt pose parameters
    """

    points_flat = points.reshape(-1, 4)
    jacobian = se3_point_jac_cuda(points_flat)

    return jacobian.reshape(*points.shape[:-1], 4, 6)


def pose_retraction(pose : Pose, tangent : torch.Tensor) -> Pose:
    other = tangent_to_pose(tangent)
    return pose_mul(pose, other)