#!/usr/bin/env python3

import os
import sys
import time
import torch
import lietorch

for p in ["/workspace", "/workspace/Splat-SLAM"]:
    if p not in sys.path:
        sys.path.append(p)

from originals import MoBA as mo_ba_old
from new import motion_only_ba as mo_ba_new

from pose_utils import Pose, Intrinsics, matrix_to_quat_cuda


def run_case(T=4, H=64, W=64, motion_scale=0.05, iters=50, seed=123, device="cuda"):
    torch.manual_seed(seed)
    device = torch.device(device)
    dtype = torch.float32

    # Edges
    if T < 2:
        E = 1
        ii = torch.tensor([0], device=device, dtype=torch.long)
        jj = torch.tensor([0], device=device, dtype=torch.long)
    else:
        ii = torch.arange(0, T - 1, device=device, dtype=torch.long)
        jj = torch.arange(1, T, device=device, dtype=torch.long)
        E = T -1 

    # Data
    target = torch.rand(E, H, W, 2, device=device, dtype=dtype)
    weight = torch.rand(E, H, W, 2, device=device, dtype=dtype)
    eta = torch.rand(T, H, W, device=device, dtype=dtype)
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
    
    target_b = target[None]
    weight_b = weight[None]
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
    intrinsics= Intrinsics(fx, fy, cx, cy)
    

    
    with torch.inference_mode():
        for _ in range(50):
            eta_b = .2 * eta[torch.unique(ii)].contiguous()[None] + 1e-7
            _ = mo_ba_old(target_b, weight_b, eta_b, poses_b, disps_b, intr_b, ii, jj)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            eta_b = .2 * eta[torch.unique(ii)][None].contiguous() + 1e-7
            old_updated_poses = mo_ba_old(target_b, weight_b, eta_b, poses_b, disps_b, intr_b, ii, jj)
            torch.cuda.synchronize()
        old_latency = (time.perf_counter() - start) * 1000.0 / iters

    with torch.inference_mode():
        for _ in range(50):
            _ = mo_ba_new(target, weight, poses_cuda, disps, intrinsics, ii, jj)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            new_updated_poses = mo_ba_new(target, weight, poses_cuda, disps, intrinsics, ii, jj)
            torch.cuda.synchronize()
        new_latency = (time.perf_counter() - start) * 1000.0 / iters

    old_vec = old_updated_poses.vec()[0]  # Remove batch dimension [N, 7]
    new_vec = torch.cat([new_updated_poses.t, new_updated_poses.q], dim=-1)  # [N, 7]
    
    updated_poses_max_diff = (old_vec - new_vec).abs().max().item()
    speedup = old_latency / new_latency
    

    stats = {
        "Updated Poses Diff": updated_poses_max_diff,
        "Old Latency (ms)": old_latency,
        "New Latency (ms)": new_latency,
        "Speedup": speedup,
    }
    
    return stats


if __name__ == "__main__":
    import pprint

    cases = [
        # Quick sanity
        dict(T=4, H=32, W=32, motion_scale=0.05, iters=1000, seed=1),
        # Typical
        dict(T=4, H=64, W=64, motion_scale=0.05, iters=1000, seed=123),
        # Low motion
        dict(T=5, H=64, W=64, motion_scale=0.01, iters=1000, seed=2),
        # High motion
        dict(T=5, H=64, W=64, motion_scale=0.2, iters=1000, seed=3),
        # Minimal edges
        dict(T=2, H=64, W=64, motion_scale=0.1, iters=1000, seed=4),
        # Larger res
        dict(T=5, H=128, W=128, motion_scale=0.1, iters=100, seed=5),
        # Wide aspect
        dict(T=4, H=64, W=128, motion_scale=0.1, iters=100, seed=6),
    ]

    for i, cfg in enumerate(cases):
        print(f"Case {i+1}:")
        print("  Config:")
        pprint.pprint(cfg, indent=4)
        stats = run_case(**cfg)
        print("  Stats:")
        pprint.pprint(stats, indent=4, sort_dicts=False)
        print()
