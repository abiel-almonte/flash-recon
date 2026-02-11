"""ATE evaluation for refactored SLAM on TUM fr1_desk.

Same as eval_ate.py but:
  - Imports from flash_recon.slam.refactored
  - Uses hybrid-compatible config
  - Upscales mono_depth to full res (matching reference pipeline)
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation

from flash_recon.slam import SLAM
from neural import MonoDepth

# TUM fr1_desk: native 640x480, crop 8px edges, resize to 512x384

H_OUT, W_OUT = 384, 512
H_EDGE, W_EDGE = 8, 8
FX_NATIVE, FY_NATIVE = 517.3, 516.5
CX_NATIVE, CY_NATIVE = 318.6, 255.3

H_CROP = 480 - 2 * H_EDGE
W_CROP = 640 - 2 * W_EDGE
FX = FX_NATIVE * W_OUT / W_CROP
FY = FY_NATIVE * H_OUT / H_CROP
CX = (CX_NATIVE - W_EDGE) * W_OUT / W_CROP
CY = (CY_NATIVE - H_EDGE) * H_OUT / H_CROP

DATASET_ROOT = os.environ.get("DATASET_ROOT", "/workspace/datasets/desk")
WEIGHTS_PATH = os.environ.get("DROID_WEIGHTS", "/workspace/neural/weights/droid.pth")
DEPTH_WEIGHTS = os.environ.get(
    "DEPTH_WEIGHTS", "/workspace/neural/weights/depth_anything_v2_vits.pth"
)
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))

cfg = {
    "device": "cuda:0",
    "weights": {"droid": WEIGHTS_PATH, "depth": DEPTH_WEIGHTS},
    "cam": {
        "H_out": H_OUT,
        "W_out": W_OUT,
        "down_scale": 8,
        "fx": FX,
        "fy": FY,
        "cx": CX,
        "cy": CY,
    },
    "tracking": {
        "beta": 0.75,
        "warmup": 12,
        "buffer": 384,
        "max_age": 50,
        "max_factors": 75,
        "mono_thres": 0.1,
        "depth_filter": {"thresh": 0.01, "n_views": 2},
        "local": {
            "nms": 1,
            "window": 25,
            "thresh": 16.0,
            "radius": 2,
            "max_factors": 75,
            "keyframe_thresh": 4.0,
            "enable_loop": True,
        },
        "loop_closure": {
            "window": 25,
            "thresh": 25.0,
            "radius": 1,
            "nms": 12,
        },
        "global": {
            "freq": 20,
            "thresh": 25.0,
            "radius": 1,
            "nms": 5,
        },
        "motion_filter": {
            "thresh": 4,
        },
    },
    "corrblock": {"num_levels": 4, "radius": 3, "max_slots": 128},
}


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_tum_rgb_list(root):
    rgb_txt = os.path.join(root, "rgb.txt")
    entries = []
    with open(rgb_txt) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            ts, path = line.split()
            entries.append((float(ts), os.path.join(root, path)))
    entries.sort(key=lambda x: x[0])
    return entries


def load_tum_groundtruth(root):
    gt_txt = os.path.join(root, "groundtruth.txt")
    entries = []
    with open(gt_txt) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            entries.append((ts, tx, ty, tz, qx, qy, qz, qw))
    entries.sort(key=lambda x: x[0])
    return entries


def associate_timestamps(rgb_timestamps, gt_entries, max_diff=0.02):
    gt_ts = np.array([e[0] for e in gt_entries])
    associations = []
    for rgb_ts in rgb_timestamps:
        idx = np.argmin(np.abs(gt_ts - rgb_ts))
        if abs(gt_ts[idx] - rgb_ts) < max_diff:
            associations.append(gt_entries[idx])
        else:
            associations.append(None)
    return associations


def gt_to_matrix(entry):
    _, tx, ty, tz, qx, qy, qz, qw = entry
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def load_and_preprocess(path, device):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if H_EDGE > 0 or W_EDGE > 0:
        img = img[H_EDGE:-H_EDGE, W_EDGE:-W_EDGE]
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.resize((W_OUT, H_OUT), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    frame = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return (frame - MEAN.to(device)) / STD.to(device)


def align_umeyama(src, dst):
    assert src.shape == dst.shape
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = np.sum(src_c**2) / n
    H = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - s * R @ mu_src
    return s, R, t


def compute_ate(est_positions, gt_positions):
    s, R, t = align_umeyama(est_positions, gt_positions)
    aligned = s * (R @ est_positions.T).T + t
    errors = np.linalg.norm(aligned - gt_positions, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
        "n_keyframes": len(errors),
        "scale": float(s),
    }



def main():
    device = cfg["device"]

    rgb_list = load_tum_rgb_list(DATASET_ROOT)
    gt_entries = load_tum_groundtruth(DATASET_ROOT)
    n_frames = MAX_FRAMES if MAX_FRAMES > 0 else len(rgb_list)
    rgb_list = rgb_list[:n_frames]
    print(f"Dataset: {DATASET_ROOT} ({len(rgb_list)} frames)")

    rgb_timestamps = [ts for ts, _ in rgb_list]
    gt_associations = associate_timestamps(rgb_timestamps, gt_entries)
    n_associated = sum(1 for g in gt_associations if g is not None)
    print(f"GT associations: {n_associated}/{len(rgb_list)} frames")

    depth_model = MonoDepth(cfg).to(device).eval()
    print("DepthAnythingV2 loaded")

    slam = SLAM(cfg)
    print("Refactored SLAM initialized")

    keyframe_frame_indices = []
    times = []

    try:
        for i, (_, path) in enumerate(rgb_list):
            frame = load_and_preprocess(path, device)

            prev_count = len(slam.buffer)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            mono_depth = depth_model(frame)
            slam(frame, mono_depth)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)

            if len(slam.buffer) > prev_count:
                keyframe_frame_indices.append(i)

            if i % 50 == 0:
                n_edges = len(slam)
                print(
                    f"  frame {i:4d}/{n_frames} | kf: {len(slam.buffer)} | "
                    f"edges: {n_edges} | {dt*1000:.0f}ms"
                )
                alloc = torch.cuda.memory_allocated() / 1e6
                reserved = torch.cuda.memory_reserved() / 1e6
                n_inac = slam.graph.ii_inac.numel()
                print(
                    f"    GPU: {alloc:.0f}MB alloc, {reserved:.0f}MB reserved | "
                    f"inac: {n_inac}"
                )

    except torch.OutOfMemoryError:
        print(f"\n  OOM at frame {i}")

    n_kf = len(slam.buffer)
    print(f"\nProcessed {len(times)} frames, {n_kf} keyframes")
    if times:
        p90 = np.percentile(times, 80) * 1000
        print(f"Timing: avg {np.mean(times)*1000:.0f}ms, median {np.median(times)*1000:.0f}ms, p80 {p90:.0f}ms")

    # Extract trajectory
    est_positions = []
    gt_positions = []
    for kf_idx, frame_idx in enumerate(keyframe_frame_indices):
        if kf_idx >= n_kf:
            break
        gt_entry = gt_associations[frame_idx]
        if gt_entry is None:
            continue
        c2w = slam.buffer.get_cam2world(kf_idx).cpu().numpy()
        est_positions.append(c2w[:3, 3])
        gt_mat = gt_to_matrix(gt_entry)
        gt_positions.append(gt_mat[:3, 3])

    est_positions = np.array(est_positions)
    gt_positions = np.array(gt_positions)

    print(f"\nATE evaluation: {len(est_positions)} keyframes with GT")

    if len(est_positions) < 3:
        print("Not enough keyframes with GT — cannot compute ATE")
        return

    est_travel = np.linalg.norm(np.diff(est_positions, axis=0), axis=1).sum()
    gt_travel = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1).sum()
    print(f"  Est trajectory: travel={est_travel:.3f}")
    print(f"  GT  trajectory: travel={gt_travel:.3f}")

    ate = compute_ate(est_positions, gt_positions)
    print(f"  ATE RMSE:   {ate['rmse']:.4f} m")
    print(f"  ATE mean:   {ate['mean']:.4f} m")
    print(f"  ATE median: {ate['median']:.4f} m")
    print(f"  ATE max:    {ate['max']:.4f} m")
    print(f"  Scale:      {ate['scale']:.8f}")
    print(f"  Keyframes:  {ate['n_keyframes']}")

    if np.any(np.isnan(est_positions)):
        print("\nFAILED — NaN in trajectory")
    elif ate["rmse"] > 1.0:
        print(f"\nWARN — ATE RMSE {ate['rmse']:.4f}m is high (>1m)")
    elif ate["rmse"] < 0.10:
        print(f"\nGOOD — ATE RMSE {ate['rmse']:.4f}m (target: 0.025m)")
    else:
        print(f"\nMIXED — ATE RMSE {ate['rmse']:.4f}m")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
