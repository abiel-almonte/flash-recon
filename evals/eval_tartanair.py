import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from flash_recon.slam import SLAM

H_OUT, W_OUT = 384, 512
FX = 320.0 * 0.8
FY = 320.0 * 0.8
CX = 320.0 * 0.8
CY = 240.0 * 0.8

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MONO_SCENES = [f"M{s}{i:03d}" for s in ["E", "H"] for i in range(8)]


def load_images(datapath, sequence):
    img_dir = os.path.join(datapath, sequence, "image_left")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(datapath, sequence)
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No images in {img_dir}")
    return paths


def load_groundtruth(gt_path, sequence):
    """Load TartanAir GT poses (NED frame) and convert to XYZ."""
    gt_file = os.path.join(gt_path, f"{sequence}.txt")
    traj = np.loadtxt(gt_file)
    # NED -> XYZ: reorder [ty, tz, tx, qy, qz, qx, qw]
    traj = traj[:, [1, 2, 0, 4, 5, 3, 6]]
    positions = traj[:, :3]
    quats_xyzw = traj[:, 3:]
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]

    timestamps = np.arange(len(positions), dtype=np.float64)
    return timestamps, positions, quats_wxyz


def evaluate_ate(est_timestamps, est_positions, est_quats_wxyz,
                 gt_positions, gt_quats_wxyz):
    from evo.core.metrics import PoseRelation
    from evo.core.trajectory import PosePath3D
    import evo.main_ape as main_ape

    # Direct index: est_timestamps are frame indices, use them to subsample GT
    kf_indices = est_timestamps.astype(int)
    gt_pos_sub = gt_positions[kf_indices]
    gt_quats_sub = gt_quats_wxyz[kf_indices]

    traj_est = PosePath3D(
        positions_xyz=est_positions,
        orientations_quat_wxyz=est_quats_wxyz,
    )
    traj_ref = PosePath3D(
        positions_xyz=gt_pos_sub,
        orientations_quat_wxyz=gt_quats_sub,
    )

    result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="flash-recon",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    return result


def pose_matrix_to_pos_quat_wxyz(c2w):
    pos = c2w[:3, 3]
    quat_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz


def build_cfg(args):
    return {
        "device": "cuda:0",
        "weights": {"droid": "neural/weights/droid.pth", "depth": ""},
        "cam": {
            "H_out": H_OUT, "W_out": W_OUT, "down_scale": 8,
            "fx": FX, "fy": FY, "cx": CX, "cy": CY,
        },
        "tracking": {
            "beta": 0.75,
            "warmup": 12,
            "buffer": 512,
            "max_age": 50,
            "max_factors": 75,
            "mono_thres": 0.1,
            "depth_filter": {"thresh": 0.01, "n_views": 2},
            "local": {
                "nms": 1, "window": 25, "thresh": 16.0, "radius": 2,
                "max_factors": 75, "keyframe_thresh": 3.0, "enable_loop": True,
            },
            "loop_closure": {
                "window": 25, "thresh": 25.0, "radius": 1, "nms": 10,
            },
            "global": {
                "freq": 20, "thresh": 25.0, "radius": 1, "nms": 5,
                "normalize": False, "enabled": False,
            },
            "motion_filter": {"thresh": 4},
        },
        "corrblock": {"num_levels": 4, "radius": 3, "max_slots": 128},
    }


def run_sequence(cfg, paths, device):
    slam = SLAM(cfg)
    times = []

    try:
        for i, path in enumerate(paths):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W_OUT, H_OUT))
            tensor = torch.from_numpy(img).float().to(device) / 255.0
            frame = tensor.permute(2, 0, 1).unsqueeze(0)
            frame = (frame - MEAN.to(device)) / STD.to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            slam(frame, tstamp=float(i))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

            if i % 200 == 0:
                print(f"  frame {i:4d}/{len(paths)} | kf: {len(slam.buffer)}")
    except torch.OutOfMemoryError:
        print(f"  OOM at frame {i}")

    print(f"  {len(times)} frames, {len(slam.buffer)} keyframes")

    print("  Terminal global BA...")
    torch.cuda.empty_cache()
    slam.finalize(steps_per_pass=(7, 12))

    n_kf = len(slam.buffer)
    est_timestamps = []
    est_positions = []
    est_quats_wxyz = []
    for kf_idx in range(n_kf):
        tstamp = slam.buffer.get_tstamp(kf_idx)
        c2w = slam.buffer.get_cam2world(kf_idx).cpu().numpy()
        pos, quat = pose_matrix_to_pos_quat_wxyz(c2w)
        est_timestamps.append(float(tstamp))
        est_positions.append(pos)
        est_quats_wxyz.append(quat)

    return (
        np.array(est_timestamps),
        np.array(est_positions),
        np.array(est_quats_wxyz),
        np.array(times),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="datasets/mono")
    parser.add_argument("--gt_path", default="datasets/mono/mono_gt")
    parser.add_argument("--scene", type=str, default=None)
    args = parser.parse_args()

    device = "cuda:0"
    cfg = build_cfg(args)
    scenes = [args.scene] if args.scene else MONO_SCENES

    results = []
    for scene in scenes:
        print(f"\n=== {scene} ===")
        torch.cuda.empty_cache()

        paths = load_images(args.datapath, scene)
        gt_ts, gt_pos, gt_quats = load_groundtruth(args.gt_path, scene)
        print(f"  {len(paths)} frames, {len(gt_pos)} GT poses")

        est_ts, est_pos, est_quats, times = run_sequence(cfg, paths, device)

        if len(est_pos) < 3:
            print("  Not enough keyframes")
            results.append((scene, float("nan")))
            continue

        result = evaluate_ate(est_ts, est_pos, est_quats, gt_pos, gt_quats)
        ate = result.stats["rmse"]
        print(f"  ATE RMSE: {ate:.6f} m")
        results.append((scene, ate))

    print("\n========================================")
    print("  Summary")
    print("========================================")
    print(f"{'Sequence':<20} {'ATE RMSE (m)':>12}")
    print(f"{'--------':<20} {'------------':>12}")
    valid = []
    for scene, ate in results:
        if np.isnan(ate):
            print(f"{scene:<20} {'FAIL':>12}")
        else:
            print(f"{scene:<20} {ate:>12.6f}")
            valid.append(ate)
    print(f"{'--------':<20} {'------------':>12}")
    if valid:
        print(f"{'Average':<20} {np.mean(valid):>12.6f} ({len(valid)} sequences)")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
