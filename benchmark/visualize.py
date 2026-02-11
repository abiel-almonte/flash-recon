"""Visualize refactored SLAM output: colored point cloud + trajectory PLY files."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from flash_recon.slam.refactored import SLAM
from neural import MonoDepth

from benchmark.eval_refactored import (
    cfg,
    H_OUT,
    W_OUT,
    H_EDGE,
    W_EDGE,
    DATASET_ROOT,
    FX,
    FY,
    CX,
    CY,
    load_tum_rgb_list,
    load_tum_groundtruth,
    associate_timestamps,
    gt_to_matrix,
    load_and_preprocess,
    align_umeyama,
    compute_ate,
)

OUTPUT_DIR = os.environ.get("VIS_OUTPUT", "/workspace/vis")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))
SUBSAMPLE_KF = int(os.environ.get("SUBSAMPLE_KF", "3"))


def write_ply(path, points, colors=None):
    """Write points (N,3) and optional colors (N,3) uint8 to PLY."""
    n = points.shape[0]
    has_color = colors is not None
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            line = f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}"
            if has_color:
                line += f" {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}"
            f.write(line + "\n")


def write_trajectory_ply(path, positions, color=(255, 0, 0)):
    """Write trajectory as colored points."""
    n = positions.shape[0]
    colors = np.tile(np.array(color, dtype=np.uint8), (n, 1))
    write_ply(path, positions, colors)


def backproject_depth(depth, c2w, fx, fy, cx, cy, valid_mask=None, stride=2):
    """Backproject depth map to world-space 3D points."""
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    u = u.flatten().astype(np.float64)
    v = v.flatten().astype(np.float64)

    z = depth[::stride, ::stride].flatten()

    if valid_mask is not None:
        mask = valid_mask[::stride, ::stride].flatten()
        mask = mask & (z > 0.01) & (z < 10.0) & np.isfinite(z)
    else:
        mask = (z > 0.01) & (z < 10.0) & np.isfinite(z)

    u, v, z = u[mask], v[mask], z[mask]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    pts_world = (c2w @ pts_cam.T).T[:, :3]

    pixel_coords = np.stack([u, v], axis=-1).astype(int)
    return pts_world, pixel_coords


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = cfg["device"]

    rgb_list = load_tum_rgb_list(DATASET_ROOT)
    gt_entries = load_tum_groundtruth(DATASET_ROOT)
    n_frames = MAX_FRAMES if MAX_FRAMES > 0 else len(rgb_list)
    rgb_list = rgb_list[:n_frames]
    print(f"Dataset: {DATASET_ROOT} ({len(rgb_list)} frames)")

    rgb_timestamps = [ts for ts, _ in rgb_list]
    gt_associations = associate_timestamps(rgb_timestamps, gt_entries)

    depth_model = MonoDepth(cfg).to(device).eval()
    slam = SLAM(cfg)
    print("Refactored SLAM initialized")

    keyframe_frame_indices = []
    keyframe_rgb_paths = []

    for i, (ts, path) in enumerate(rgb_list):
        frame = load_and_preprocess(path, device)

        # Mono depth: predict at low res, upscale to full (matching eval pipeline)
        mono_depth = depth_model(frame)
        mono_depth_full = F.interpolate(
            mono_depth[None, None], size=(H_OUT, W_OUT),
            mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)

        prev_count = len(slam.buffer)
        slam(frame, mono_depth_full)

        if len(slam.buffer) > prev_count:
            keyframe_frame_indices.append(i)
            keyframe_rgb_paths.append(path)

        if i % 100 == 0:
            print(f"  frame {i:4d}/{n_frames} | kf: {len(slam.buffer)}")

    n_kf = len(slam.buffer)
    print(f"\nProcessed {n_frames} frames, {n_kf} keyframes")

    # --- Extract trajectories ---
    est_positions = []
    gt_positions = []
    est_c2ws = []

    for kf_idx, frame_idx in enumerate(keyframe_frame_indices):
        if kf_idx >= n_kf:
            break
        c2w = slam.buffer.get_cam2world(kf_idx).cpu().numpy()
        est_c2ws.append(c2w)
        est_positions.append(c2w[:3, 3])

        gt_entry = gt_associations[frame_idx]
        if gt_entry is not None:
            gt_positions.append(gt_to_matrix(gt_entry)[:3, 3])
        else:
            gt_positions.append(None)

    est_pos_arr = np.array(est_positions)

    # Filter to frames with GT
    valid_gt = [
        (i, ep, gp)
        for i, (ep, gp) in enumerate(zip(est_positions, gt_positions))
        if gp is not None
    ]
    est_with_gt = np.array([ep for _, ep, _ in valid_gt])
    gt_arr = np.array([gp for _, _, gp in valid_gt])

    ate = compute_ate(est_with_gt, gt_arr)
    s, R, t = align_umeyama(est_with_gt, gt_arr)
    print(f"\nATE RMSE: {ate['rmse']:.4f}m, Scale: {s:.4f}")

    # Align full trajectory
    aligned_est = s * (R @ est_pos_arr.T).T + t

    # --- Save trajectory PLYs ---
    write_trajectory_ply(
        os.path.join(OUTPUT_DIR, "traj_est_aligned.ply"), aligned_est, color=(255, 0, 0)
    )
    write_trajectory_ply(
        os.path.join(OUTPUT_DIR, "traj_gt.ply"), gt_arr, color=(0, 255, 0)
    )
    write_trajectory_ply(
        os.path.join(OUTPUT_DIR, "traj_est_raw.ply"), est_pos_arr, color=(0, 0, 255)
    )

    # Per-frame errors
    aligned_with_gt = s * (R @ est_with_gt.T).T + t
    errors = np.linalg.norm(aligned_with_gt - gt_arr, axis=1)
    worst_idx = np.argmax(errors)
    print(
        f"Worst frame: kf_idx={valid_gt[worst_idx][0]}, error={errors[worst_idx]:.4f}m"
    )
    print(
        f"Error distribution: min={errors.min():.4f}, p25={np.percentile(errors,25):.4f}, "
        f"p50={np.percentile(errors,50):.4f}, p75={np.percentile(errors,75):.4f}, max={errors.max():.4f}"
    )

    # --- Build colored point cloud ---
    print(f"\nBuilding point cloud (every {SUBSAMPLE_KF}th keyframe)...")

    # Use aligned c2ws for the point cloud
    aligned_c2ws = []
    for c2w in est_c2ws:
        pos = c2w[:3, 3]
        aligned_pos = s * (R @ pos) + t
        ac2w = np.eye(4)
        ac2w[:3, :3] = R @ c2w[:3, :3]
        ac2w[:3, 3] = aligned_pos
        aligned_c2ws.append(ac2w)

    fx_hr, fy_hr, cx_hr, cy_hr = FX, FY, CX, CY

    all_points = []
    all_colors = []

    for kf_idx in range(0, n_kf, SUBSAMPLE_KF):
        if kf_idx >= len(keyframe_rgb_paths):
            break

        # Bilinear upsample low-res disparity to full res
        disp_lr = slam.buffer._disps[kf_idx].unsqueeze(0).unsqueeze(0)
        disp_up = (
            F.interpolate(
                disp_lr, size=(H_OUT, W_OUT), mode="bilinear", align_corners=False
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        vmask = slam.buffer._valid_depth_mask[kf_idx].cpu().numpy()

        # Convert disparity to depth (in aligned scale)
        depth = s / np.clip(disp_up, 1e-5, None)

        # Tighter depth filtering — reject far/noisy points
        depth_mask = vmask & (depth > 0.1) & (depth < 5.0)

        c2w = aligned_c2ws[kf_idx]

        # Load RGB image for coloring
        img = (
            np.array(
                Image.open(keyframe_rgb_paths[kf_idx]).convert("RGB"), dtype=np.float32
            )
            / 255.0
        )
        if H_EDGE > 0 or W_EDGE > 0:
            img = img[H_EDGE:-H_EDGE, W_EDGE:-W_EDGE]
        img = np.array(
            Image.fromarray((img * 255).astype(np.uint8)).resize(
                (W_OUT, H_OUT), Image.BILINEAR
            )
        )

        pts, pix = backproject_depth(
            depth, c2w, fx_hr, fy_hr, cx_hr, cy_hr, valid_mask=depth_mask, stride=4
        )

        if pts.shape[0] > 0:
            colors = img[pix[:, 1], pix[:, 0]]
            all_points.append(pts)
            all_colors.append(colors)

        if kf_idx % 30 == 0:
            depth_valid = depth[vmask] if vmask.any() else depth[depth > 0]
            print(
                f"  kf {kf_idx}/{n_kf}: {pts.shape[0]} pts, "
                f"depth [{depth_valid.min():.2f}, {np.median(depth_valid):.2f}, {depth_valid.max():.2f}]m, "
                f"vmask {vmask.sum()}/{vmask.size} ({100*vmask.mean():.0f}%)"
            )

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Filter outliers (points far from trajectory)
    traj_center = aligned_est.mean(axis=0)
    traj_radius = np.linalg.norm(aligned_est - traj_center, axis=1).max()
    dist_to_center = np.linalg.norm(all_points - traj_center, axis=1)
    inlier = dist_to_center < traj_radius * 1.5
    all_points = all_points[inlier]
    all_colors = all_colors[inlier]

    print(f"\nTotal points: {all_points.shape[0]:,} (after outlier removal)")

    ply_path = os.path.join(OUTPUT_DIR, "pointcloud.ply")
    write_ply(ply_path, all_points, all_colors)
    print(f"Saved: {ply_path}")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'traj_est_aligned.ply')} (red)")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'traj_gt.ply')} (green)")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'traj_est_raw.ply')} (blue)")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
