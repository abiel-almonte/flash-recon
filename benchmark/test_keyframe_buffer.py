#!/usr/bin/env python3

import sys
import torch
import lietorch

for p in ["/workspace"]:
    if p not in sys.path:
        sys.path.append(p)

from geometry import (
    Intrinsics,
    matrix_to_pose,
)
from tracker import KeyFrameBuffer, CallerRole


def make_cfg(H=240, W=320):
    return {
        "device": "cuda",
        "cam": {
            "down_scale": 8,
            "H_out": H,
            "W_out": W,
            "fx": 300.0,
            "fy": 300.0,
            "cx": W / 2.0,
            "cy": H / 2.0,
        },
        "tracking": {
            "frontend": {"radius": 1},
            "warmup": 1,
            "buffer": 64,
            "depth_filter": {"thresh": 0.01, "n_views": 2},
        },
        "optim": {"iters_per_call": 2},
    }


@torch.inference_mode()
def run_smoke(T=5, H=96, W=128, motion_scale=0.05, seed=123):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.float32

    # Buffer
    cfg = make_cfg(H=H, W=W)
    buf = KeyFrameBuffer(cfg)
    # Set intrinsics explicitly for payload paths
    fx, fy = 300.0, 300.0
    cx, cy = W / 2.0, H / 2.0
    intr = Intrinsics(fx, fy, cx, cy)
    buf.set_intrinsics(intr)

    # Poses
    xi = torch.randn(T, 6, device=device, dtype=dtype) * motion_scale
    poses_se3 = lietorch.SE3.exp(xi)
    poses = matrix_to_pose(poses_se3.matrix())

    # Disps (downsampled)
    disps = torch.rand(
        T, H // buf.down_scale, W // buf.down_scale, device=device, dtype=dtype
    ).clamp_min(1e-4)
    mono_disps = torch.rand_like(disps)

    # Append frames
    for t in range(T):
        buf.append(poses[t], disps[t], mono_disps[t])

    assert len(buf) == T

    # Update small vmask (down) -> used for scale/shift
    buf.update_vmask(up=False)
    vmask_small = buf._valid_depth_mask_small[:T]
    assert vmask_small.dtype == torch.bool and vmask_small.shape[-2:] == (
        H // buf.down_scale,
        W // buf.down_scale,
    )

    # Create DSPO payload (frontend) -> computes scales/shifts
    payload = buf.create_dspo_payload(role=CallerRole.FRONTEND)  # ensure no crash

    # Upsample a couple frames
    source_indices = torch.tensor([0, T - 1], device=device, dtype=torch.long)
    upmask = torch.randn(
        source_indices.numel(),
        9,
        8,
        8,
        H // buf.down_scale,
        W // buf.down_scale,
        device=device,
        dtype=dtype,
    )
    buf.upsample_disps(source_indices, upmask)

    # Update full-res vmask (up)
    buf.update_vmask(up=True)
    vmask_up = buf._valid_depth_mask[:T]
    assert vmask_up.dtype == torch.bool and vmask_up.shape[-2:] == (H, W)

    # Idempotence: calling update_vmask(up=True) again without changes should keep masks identical
    vmask_up_before = vmask_up.clone()
    buf.update_vmask(up=True)
    assert torch.equal(vmask_up_before, buf._valid_depth_mask[:T])

    # Normalize should mark all frames for recompute
    buf.normalize()
    assert buf.needs_update[:T].any()

    # Getters sanity
    c2w = buf.get_camera2world(0)
    depth0 = buf.get_depth(0)
    assert c2w.shape == (4, 4), c2w.shape
    assert depth0.shape == (H // buf.down_scale, W // buf.down_scale)

    # Threshold behavior: increasing required views should not increase mask density (downsampled)
    buf.update_vmask(up=False)
    base_mask = buf._valid_depth_mask_small[:T]
    base_frac = base_mask.float().mean().item()
    old_nviews = buf.depth_filter_n_views
    buf.depth_filter_n_views = old_nviews + 1
    buf.update_vmask(up=False)
    tighter_mask = buf._valid_depth_mask_small[:T]
    tighter_frac = tighter_mask.float().mean().item()
    assert tighter_frac <= base_frac + 1e-6
    buf.depth_filter_n_views = old_nviews

    # Scale/shift correctness on synthetic data
    T2 = 3
    cfg2 = make_cfg(H=H, W=W)
    buf2 = KeyFrameBuffer(cfg2)
    buf2.set_intrinsics(intr)
    a, b = 1.7, -0.3
    mono = torch.rand(T2, H // buf2.down_scale, W // buf2.down_scale, device=device, dtype=dtype)
    disps_syn = a * mono + b
    for k in range(T2):
        buf2.append(poses[k], disps_syn[k], mono[k])
    # Force weights to ones
    buf2._valid_depth_mask_small[:T2] = True
    buf2.update_scale_shift(buf2._mono_disps[:T2], buf2._disps[:T2], buf2._valid_depth_mask_small[:T2])
    est_a = buf2._scales[:T2].mean().item()
    est_b = buf2._shifts[:T2].mean().item()
    assert abs(est_a - a) < 1e-4 and abs(est_b - b) < 1e-4

    # Normalize invariants: mean(disps)=1 and translations scaled
    buf3 = KeyFrameBuffer(cfg)
    buf3.set_intrinsics(intr)
    for k in range(T):
        buf3.append(poses[k], disps[k], mono_disps[k])
    pre_mean = buf3._disps[:T].mean().item()
    pre_t = buf3._poses.t[:T].clone()
    buf3.normalize()
    post_mean = buf3._disps[:T].mean().item()
    assert abs(post_mean - 1.0) < 1e-5
    assert torch.allclose(buf3._poses.t[:T], pre_t * pre_mean, atol=1e-6)

    # Payload roles
    p_front = buf.create_dspo_payload(role=CallerRole.FRONTEND)
    p_back = buf.create_dspo_payload(role=CallerRole.BACKEND)
    p_traj = buf.create_dspo_payload(role=CallerRole.TRAJ_FILLER)
    assert p_front.mono_disps is not None and p_front.valid_depth_mask is not None
    assert p_back.mono_disps is None and p_back.valid_depth_mask is None
    assert p_traj.mono_disps is None and p_traj.valid_depth_mask is None

    # Capacity overflow
    small_cfg = make_cfg(H=H, W=W)
    small_cfg["tracking"]["buffer"] = 2
    small = KeyFrameBuffer(small_cfg)
    small.set_intrinsics(intr)
    small.append(poses[0], disps[0], mono_disps[0])
    small.append(poses[1], disps[1], mono_disps[1])
    try:
        small.append(poses[2], disps[2], mono_disps[2])
        raise AssertionError("Expected capacity overflow")
    except RuntimeError as e:
        print(e)
        pass

    print("KeyFrameBuffer extended tests passed.")


if __name__ == "__main__":
    run_smoke()
