#!/usr/bin/env python3

import os
import sys
import time
import torch
import lietorch

# Make both workspace roots importable (common in dev containers vs local)
for p in ["/workspace", "/workspace/Splat-SLAM"]:
    if p not in sys.path:
        sys.path.append(p)

from originals import projective_transform as projective_transform_old
from new import projective_transform as projective_transform_new

from pose_utils import Pose, Intrinsics, matrix_to_quat_cuda


def run_case(T=4, H=64, W=64, motion_scale=0.05, iters=50, seed=123, device="cuda"):
    torch.manual_seed(seed)
    device = torch.device(device)
    dtype = torch.float32

    # Data
    disps = torch.rand(T, H, W, device=device, dtype=dtype).clamp_min(1e-4)
    fx = 300.0
    fy = 300.0
    cx = W / 2.0
    cy = H / 2.0
    intrinsics_1d = torch.tensor([fx, fy, cx, cy], device=device, dtype=dtype)

    # Poses
    xi = torch.randn(T, 6, device=device, dtype=dtype) * motion_scale
    poses_se3 = lietorch.SE3.exp(xi)
    poses_vec = poses_se3.vec().contiguous()  # [T, 7]

    poses_b = lietorch.SE3(poses_vec[None])
    disps_b = disps[None]
    intr_b = intrinsics_1d.view(1, 1, 4).repeat(1, T, 1)

    # Edges
    if T < 2:
        ii = torch.tensor([0], device=device, dtype=torch.long)
        jj = torch.tensor([0], device=device, dtype=torch.long)
    else:
        ii = torch.arange(0, T - 1, device=device, dtype=torch.long)
        jj = torch.arange(1, T, device=device, dtype=torch.long)

    R = poses_se3.matrix()[:, :3, :3].contiguous()
    t = poses_se3.matrix()[:, :3, 3].contiguous()
    q = matrix_to_quat_cuda(R)
    poses_cuda = Pose(t, q)
    intrinsics_tuple = Intrinsics(fx, fy, cx, cy)

    with torch.inference_mode():
        for _ in range(10):
            projective_transform_old(poses_b, disps_b, intr_b, ii, jj, jacobian=True)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            old_coords, old_valid, (old_Ji, old_Jj, old_Jz) = projective_transform_old(
                poses_b, disps_b, intr_b, ii, jj, jacobian=True
            )
            torch.cuda.synchronize()
        old_latency = (time.perf_counter() - start) * 1000.0 / iters

    with torch.inference_mode():
        for _ in range(10):
            projective_transform_new(
                poses_cuda, disps, intrinsics_tuple, ii, jj, jacobian=True
            )
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            new_coords, new_valid, new_Ji, new_Jj, new_Jz = projective_transform_new(
                poses_cuda, disps, intrinsics_tuple, ii, jj, jacobian=True
            )
            torch.cuda.synchronize()
        new_latency = (time.perf_counter() - start) * 1000.0 / iters

    old_coords = old_coords.squeeze(0)
    old_valid = old_valid.squeeze(0)
    old_Ji = old_Ji.squeeze(0)
    old_Jj = old_Jj.squeeze(0)
    old_Jz = old_Jz.squeeze(0)

    valid_match_rate = (old_valid == new_valid).float().mean().item()
    both_valid = (old_valid > 0.5) & (new_valid > 0.5)  # [E, H, W, 1]

    if both_valid.any():
        coords_max_diff = (old_coords - new_coords).abs().max().item()
        ji_max_diff = (old_Ji - new_Ji).abs().max().item()
        jj_max_diff = (old_Jj - new_Jj).abs().max().item()
        jz_max_diff = (old_Jz - new_Jz).abs().max().item()
    else:
        coords_max_diff = float("nan")
        ji_max_diff = float("nan")
        jj_max_diff = float("nan")
        jz_max_diff = float("nan")

    stats = {
        "Coords Max Diff": coords_max_diff,
        "Valid Match Rate": valid_match_rate,
        "Ji Max Diff": ji_max_diff,
        "Jj Max Diff": jj_max_diff,
        "Jz Max Diff": jz_max_diff,
        "Old Latency (ms)": old_latency,
        "New Latency (ms)": new_latency,
        "Speedup x": old_latency / new_latency,
    }

    return stats


if __name__ == "__main__":
    import pprint

    cases = [
        # Quick sanity
        dict(T=3, H=32, W=32, motion_scale=0.05, iters=200, seed=1),
        # Typical
        dict(T=4, H=64, W=64, motion_scale=0.05, iters=200, seed=123),
        # Low motion
        dict(T=5, H=64, W=64, motion_scale=0.01, iters=200, seed=2),
        # High motion
        dict(T=5, H=64, W=64, motion_scale=0.2, iters=200, seed=3),
        # Minimal edges
        dict(T=2, H=64, W=64, motion_scale=0.1, iters=200, seed=4),
        # Larger res
        dict(T=5, H=128, W=128, motion_scale=0.1, iters=100, seed=5),
        # Wide aspect
        dict(T=4, H=64, W=128, motion_scale=0.1, iters=100, seed=6),
        # Stress
        dict(T=8, H=240, W=320, motion_scale=0.1, iters=50, seed=7),
    ]

    for i, cfg in enumerate(cases):
        print(f"Case {i+1}:")
        print("  Config:")
        pprint.pprint(cfg, indent=4)
        stats = run_case(**cfg)
        print("  Stats:")
        pprint.pprint(stats, indent=4, sort_dicts=False)
        print()
