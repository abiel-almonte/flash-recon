#!/usr/bin/env python3

import os
import sys
import time
import torch
import lietorch

for p in ["/workspace", "/workspace/Splat-SLAM"]:
    if p not in sys.path:
        sys.path.append(p)

from originals import BA_with_scale_shift as ba_ss_old
from new import ba_scale_shift as ba_ss_new

from geometry import Intrinsics, matrix_to_pose


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
        E = T - 1

    # Data
    target = torch.rand(E, H, W, 2, device=device, dtype=dtype)
    weight = torch.rand(E, H, W, 2, device=device, dtype=dtype)
    eta = torch.rand(T, H, W, device=device, dtype=dtype)
    disps = torch.rand(T, H, W, device=device, dtype=dtype).clamp_min(1e-4)
    mono_disps = torch.rand(T, H, W, device=device, dtype=dtype).clamp_min(1e-4)
    scales = torch.rand(T, device=device, dtype=dtype)
    shifts = torch.rand(T, device=device, dtype=dtype)
    valid_depth_mask = torch.randint(0, 1, (T, H, W), device=device, dtype=torch.bool)

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
    mono_disps_b = mono_disps[None]
    scales_b = scales[None]
    shifts_b = shifts[None]
    valid_depth_mask_b = valid_depth_mask[None]

    # Edges
    if T < 2:
        ii = torch.tensor([0], device=device, dtype=torch.long)
        jj = torch.tensor([0], device=device, dtype=torch.long)
    else:
        ii = torch.arange(0, T - 1, device=device, dtype=torch.long)
        jj = torch.arange(1, T, device=device, dtype=torch.long)

    poses_cuda = matrix_to_pose(poses_se3.matrix())
    intrinsics = Intrinsics(fx, fy, cx, cy)

    with torch.inference_mode():
        for _ in range(50):
            eta_b = 0.2 * eta[torch.unique(ii)].contiguous()[None] + 1e-7
            ba_ss_old(
                target_b,
                weight_b,
                eta_b,
                poses_b,
                disps_b,
                intr_b,
                ii,
                jj,
                mono_disps_b,
                scales_b,
                shifts_b,
                valid_depth_mask_b,
            )
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            eta_b = 0.2 * eta[torch.unique(ii)][None].contiguous() + 1e-7
            _, old_updated_disps, old_wqs = ba_ss_old(
                target_b,
                weight_b,
                eta_b,
                poses_b,
                disps_b,
                intr_b,
                ii,
                jj,
                mono_disps_b,
                scales_b,
                shifts_b,
                valid_depth_mask_b,
            )
            torch.cuda.synchronize()
        old_latency = (time.perf_counter() - start) * 1000.0 / iters

    eta_keyframes = eta[torch.unique(ii)]

    with torch.inference_mode():
        for _ in range(50):
            new_updated_disps, new_wqs = ba_ss_new(
                target,
                weight,
                eta_keyframes,
                poses_cuda,
                disps,
                intrinsics,
                ii,
                jj,
                mono_disps,
                scales,
                shifts,
                valid_depth_mask,
            )
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            new_updated_disps, new_wqs = ba_ss_new(
                target,
                weight,
                eta_keyframes,
                poses_cuda,
                disps,
                intrinsics,
                ii,
                jj,
                mono_disps,
                scales,
                shifts,
                valid_depth_mask,
            )
            torch.cuda.synchronize()
        new_latency = (time.perf_counter() - start) * 1000.0 / iters

    old_updated_disps = old_updated_disps.squeeze(0)
    old_wqs = old_wqs.squeeze(0)

    old_updated_disps_max_diff = (
        (old_updated_disps - new_updated_disps).abs().max().item()
    )
    old_wqs_max_dif = (old_wqs - new_wqs).abs().max().item()

    stats = {
        "Optimized Disps Max Diff": old_updated_disps_max_diff,
        "WQS Max Diff": old_wqs_max_dif,
        "Old Latency (ms)": old_latency,
        "New Latency (ms)": new_latency,
        "Speedup x": old_latency / new_latency,
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
