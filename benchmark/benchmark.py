import torch
import time
import lietorch

import sys
sys.path.insert(0, "/workspace")

from originals import update_pose as update_pose_original
from originals import projective_transform as projective_transform_original

from new import update_pose as update_pose_new
from new import projective_transform as projective_transform_new
from pose_utils import Pose, Intrinsics

class MockCamera:
    def __init__(self, device="cuda"):
        self.R = torch.eye(3, device=device, dtype=torch.float32)
        self.T = torch.zeros(3, device=device, dtype=torch.float32)
        self.cam_trans_delta = torch.randn(3, device=device, dtype=torch.float32) * 0.01
        self.cam_rot_delta = torch.randn(3, device=device, dtype=torch.float32) * 0.01
        
    def update_RT(self, R, T):
        self.R = R.clone()
        self.T = T.clone()

def time_func(func, n=1000):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        func()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n * 1000

def benchmark_pose_update():
    camera_orig = MockCamera()
    camera_cuda = MockCamera()
    
    camera_cuda.R = camera_orig.R.clone()
    camera_cuda.T = camera_orig.T.clone() 
    camera_cuda.cam_trans_delta = camera_orig.cam_trans_delta.clone()
    camera_cuda.cam_rot_delta = camera_orig.cam_rot_delta.clone()
    
    orig_time = time_func(lambda: update_pose_original(camera_orig))
    cuda_time = time_func(lambda: update_pose_new(camera_cuda))
    
    print(f"Pose Update:")
    print(f"  Original: {orig_time:.3f}ms")
    print(f"  CUDA:     {cuda_time:.3f}ms")
    print(f"  Speedup:  {orig_time/cuda_time:.1f}x")

def benchmark_projective_transform():
    device = "cuda"
    B, N_frames, H, W = 1, 3, 64, 64
    
    orig_depths = torch.randn(B, N_frames, H, W, device=device, dtype=torch.float32).abs() + 0.5
    orig_intrinsics = torch.tensor([320.0, 320.0, 32.0, 32.0], device=device).unsqueeze(0).unsqueeze(0).repeat(B, N_frames, 1)
    xi = torch.randn(B, N_frames, 6, device=device, dtype=torch.float32) * 0.1
    orig_poses = lietorch.SE3.exp(xi)
    
    cuda_depths = orig_depths[0]  # [N_frames, H, W]
    cuda_intrinsics = Intrinsics(320.0, 320.0, 32.0, 32.0)
    
    pose_t = torch.randn(N_frames, 3, device=device, dtype=torch.float32) * 0.1
    pose_q = torch.randn(N_frames, 4, device=device, dtype=torch.float32)
    pose_q = pose_q / pose_q.norm(dim=1, keepdim=True)
    cuda_poses = Pose(pose_t, pose_q)
    
    ii = torch.tensor([0], device=device)
    jj = torch.tensor([1], device=device)
    
    orig_time = time_func(lambda: projective_transform_original(orig_poses, orig_depths, orig_intrinsics, ii, jj, jacobian=True), n=1000)
    cuda_time = time_func(lambda: projective_transform_new(cuda_poses, cuda_depths, cuda_intrinsics, ii, jj, jacobian=True), n=1000)
    
    print(f"Projective Transform:")
    print(f"  Original: {orig_time:.3f}ms")
    print(f"  CUDA:     {cuda_time:.3f}ms")
    print(f"  Speedup:  {orig_time/cuda_time:.1f}x")

if __name__ == "__main__":
    print("Optimizations for Splat-SLAM")
    print("=" * 40)
    print()
    
    benchmark_pose_update()
    print()
    benchmark_projective_transform()