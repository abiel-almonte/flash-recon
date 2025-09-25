import torch
import time
import lietorch

import sys

sys.path.insert(0, "/workspace")

from originals import update_pose as update_pose_original
from originals import (
    projective_transform as projective_transform_original,
    induced_flow as induced_flow_original,
)

from new import update_pose as update_pose_new
from new import (
    projective_transform as projective_transform_new,
    induced_flow as induced_flow_new,
)
from projective_utils.src.ops import projective_transform_fused
from pose_utils import Pose, Intrinsics
from pose_utils.src.ops import matrix_to_quat_cuda

import warnings

warnings.filterwarnings(
    "ignore", "torch.meshgrid: in an upcoming release"
)  # for induced_flow_original


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

    orig_time = time_func(lambda: update_pose_original(camera_orig), n=2000)
    cuda_time = time_func(lambda: update_pose_new(camera_cuda), n=2000)

    print(f"Pose Update:")
    print(f"  Original: {orig_time:.3f}ms")
    print(f"  CUDA:     {cuda_time:.3f}ms")
    print(f"  Speedup:  {orig_time/cuda_time:.1f}x")


def benchmark_projective_transform():
    device = "cuda"
    B, N_frames, H, W = 1, 5, 64, 64

    orig_depths = (
        torch.randn(B, N_frames, H, W, device=device, dtype=torch.float32).abs() + 0.5
    )
    orig_intrinsics = (
        torch.tensor([320.0, 320.0, 32.0, 32.0], device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(B, N_frames, 1)
    )
    xi = torch.randn(B, N_frames, 6, device=device, dtype=torch.float32) * 0.1
    orig_poses = lietorch.SE3.exp(xi)

    # Use the same poses/depths/intrinsics across all paths
    pose_mats = orig_poses[0].matrix()
    pose_t = pose_mats[:, :3, 3]
    pose_R = pose_mats[:, :3, :3].contiguous()
    pose_q = matrix_to_quat_cuda(pose_R)
    cuda_poses = Pose(pose_t, pose_q)
    cuda_depths = orig_depths[0]  # [N_frames, H, W]
    intr_o = orig_intrinsics[0, 0]
    cuda_intrinsics = Intrinsics(
        intr_o[0].item(), intr_o[1].item(), intr_o[2].item(), intr_o[3].item()
    ).to("cuda")

    ii = torch.tensor([0], device=device)
    jj = torch.tensor([1], device=device)

    orig_time = time_func(
        lambda: projective_transform_original(
            orig_poses, orig_depths, orig_intrinsics, ii, jj, jacobian=False
        ),
        n=2000,
    )
    cuda_time = time_func(
        lambda: projective_transform_new(
            cuda_poses, cuda_depths, cuda_intrinsics, ii, jj, jacobian=False
        ),
        n=2000,
    )
    # fused_time = time_func(lambda: projective_transform_fused(cuda_poses, cuda_depths, cuda_intrinsics, ii, jj), n=1000)

    print(f"Projective Transform:")
    print(f"  Original: {orig_time:.3f}ms")
    print(f"  CUDA:     {cuda_time:.3f}ms")
    # print(f"  Fused:    {fused_time:.3f}ms")
    print(
        f"  Speedup:  {orig_time/cuda_time:.1f}x vs CUDA"
    )  # {orig_time/fused_time:.1f}x vs Fused")

    # Correctness check (coords/valid)
    with torch.no_grad():
        coords_o, valid_o = projective_transform_original(
            orig_poses, orig_depths, orig_intrinsics, ii, jj
        )
        coords_o = coords_o.squeeze(0)[..., :2]
        valid_o = valid_o.squeeze(0)
        coords_c, valid_c = projective_transform_new(
            cuda_poses, cuda_depths, cuda_intrinsics, ii, jj, jacobian=False
        )

        # Also compare fused vs Original (sanity)
        diff_all_o = (coords_c - coords_o).abs()
        both_valid_o = (valid_c > 0.5) & (valid_o > 0.5)
        if both_valid_o.any():
            mask_o = both_valid_o.expand_as(diff_all_o)[..., 0]
            diff_valid_o = diff_all_o[mask_o]
            max_vo = diff_valid_o.max().item()
            mean_vo = diff_valid_o.mean().item()
            std_vo = diff_valid_o.std().item()
        else:
            max_vo = float("nan")
            mean_vo = float("nan")
            std_vo = float("nan")

        vmr_fo = (valid_c == valid_o).float().mean().item()

    print(
        f"  Correctness (Fused vs Orig)   max/mean/std: {max_vo:.2e}/{mean_vo:.2e}/{std_vo:.2e}, valid_match: {vmr_fo:.4f}"
    )


def benchmark_induced_flow():
    device = "cuda"
    B, N_frames, H, W = 1, 5, 64, 64

    orig_depths = (
        torch.randn(B, N_frames, H, W, device=device, dtype=torch.float32).abs() + 0.5
    )
    orig_intrinsics = (
        torch.tensor([320.0, 320.0, 32.0, 32.0], device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(B, N_frames, 1)
    )
    xi = torch.randn(B, N_frames, 6, device=device, dtype=torch.float32) * 0.1
    orig_poses = lietorch.SE3.exp(xi)

    cuda_depths = orig_depths[0]  # [N_frames, H, W]
    cuda_intrinsics = Intrinsics(320.0, 320.0, 32.0, 32.0).to("cuda")

    pose_t = torch.randn(N_frames, 3, device=device, dtype=torch.float32) * 0.1
    pose_q = torch.randn(N_frames, 4, device=device, dtype=torch.float32)
    pose_q = pose_q / pose_q.norm(dim=1, keepdim=True)
    cuda_poses = Pose(pose_t, pose_q)

    ii = torch.tensor([0], device=device)
    jj = torch.tensor([1], device=device)

    orig_time = time_func(
        lambda: induced_flow_original(orig_poses, orig_depths, orig_intrinsics, ii, jj),
        n=1000,
    )
    cuda_time = time_func(
        lambda: induced_flow_new(cuda_poses, cuda_depths, cuda_intrinsics, ii, jj),
        n=1000,
    )

    print(f"Induced Flow:")
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
    print()
    benchmark_induced_flow()
    print()
