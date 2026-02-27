"""EuRoC MAV dataset evaluation for flash-recon SLAM."""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from flash_recon.slam import SLAM

# ---------------------------------------------------------------------------
# EuRoC left camera raw calibration (from sensor.yaml)
# ---------------------------------------------------------------------------
K_L = np.array([458.654, 0.0, 367.215,
                0.0, 457.296, 248.375,
                0.0, 0.0, 1.0]).reshape(3, 3)
D_L = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
R_L = np.array([
    0.999966347530033, -0.001422739138722922, 0.008079580483432283,
    0.001365741834644127, 0.9999741760894847, 0.007055629199258132,
    -0.008089410156878961, -0.007044357138835809, 0.9999424675829176,
]).reshape(3, 3)
P_L = np.array([
    435.2046959714599, 0, 367.4517211914062, 0,
    0, 435.2046959714599, 252.2008514404297, 0,
    0, 0, 1, 0,
]).reshape(3, 4)

# Rectified intrinsics (after undistortion)
FX_RECT = 435.2046959714599
FY_RECT = 435.2046959714599
CX_RECT = 367.4517211914062
CY_RECT = 252.2008514404297
HT_NATIVE, WD_NATIVE = 480, 752

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_undistort_map():
    """Build OpenCV remap tables for EuRoC left camera undistortion+rectification."""
    return cv2.initUndistortRectifyMap(
        K_L, D_L, R_L, P_L[:3, :3], (WD_NATIVE, HT_NATIVE), cv2.CV_32F
    )


def resolve_euroc_datapath(datapath):
    """Resolve EuRoC datapath, handling the grouped archive layout.

    The ETH Research Collection archives extract to:
        datasets/EuRoC/machine_hall/MH_01_easy/mav0/...
        datasets/EuRoC/vicon_room1/V1_01_easy/mav0/...
        datasets/EuRoC/vicon_room2/V2_01_easy/mav0/...

    But users may pass datasets/EuRoC/MH_01_easy (flat layout).
    This function checks both and returns the path that has images.
    """
    if os.path.isdir(os.path.join(datapath, "mav0")):
        return datapath

    # Try grouped layout: infer subfolder from sequence name
    scene = os.path.basename(os.path.normpath(datapath))
    parent = os.path.dirname(os.path.normpath(datapath))
    if scene.startswith("MH"):
        grouped = os.path.join(parent, "machine_hall", scene)
    elif scene.startswith("V1"):
        grouped = os.path.join(parent, "vicon_room1", scene)
    elif scene.startswith("V2"):
        grouped = os.path.join(parent, "vicon_room2", scene)
    else:
        grouped = datapath

    if os.path.isdir(os.path.join(grouped, "mav0")):
        return grouped

    return datapath  # fall through, will error later with a clear message


class EuRoCImageStream:
    """Lazy image loader for EuRoC sequences — loads one frame at a time."""

    def __init__(self, datapath, image_size, stride=1, max_frames=0):
        datapath = resolve_euroc_datapath(datapath)
        self.map_x, self.map_y = build_undistort_map()

        image_paths = sorted(
            glob.glob(os.path.join(datapath, "mav0/cam0/data/*.png"))
        )
        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {datapath}/mav0/cam0/data/\n"
                f"If using the grouped ETH archive, ensure the per-sequence zip is extracted.\n"
                f"Try: python3 -c \"import zipfile; zipfile.ZipFile('<path>/MH_01_easy.zip').extractall('<path>/')\""
            )

        # Build (timestamp_ns, path) list, apply stride
        all_entries = [
            (float(os.path.basename(p)[:-4]), p) for p in image_paths
        ]
        selected = all_entries[::stride]
        if max_frames > 0:
            selected = selected[:max_frames]

        self.entries = selected  # [(tstamp_ns, path), ...]
        self.ht_out, self.wd_out = image_size

        # Scale intrinsics to target resolution
        self.intrinsics = (
            FX_RECT * self.wd_out / WD_NATIVE,
            FY_RECT * self.ht_out / HT_NATIVE,
            CX_RECT * self.wd_out / WD_NATIVE,
            CY_RECT * self.ht_out / HT_NATIVE,
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        """Load and preprocess a single frame. Returns (tstamp_ns, tensor[1,3,H,W])."""
        tstamp_ns, img_path = self.entries[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Failed to read {img_path}")

        img = cv2.remap(img, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (self.wd_out, self.ht_out))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - MEAN) / STD
        return tstamp_ns, tensor.unsqueeze(0)

    def get_timestamp(self, idx):
        """Get timestamp without loading the image."""
        return self.entries[idx][0]


def load_euroc_groundtruth(gt_path):
    """Load EuRoC ground truth file.

    Format: timestamp_ns px py pz qw qx qy qz
    Returns parallel arrays of (timestamps_s, positions, quaternions_wxyz).
    """
    timestamps = []
    positions = []
    quats_wxyz = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            ts_ns = float(parts[0])
            timestamps.append(ts_ns / 1e9)
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            quats_wxyz.append([
                float(parts[4]), float(parts[5]),
                float(parts[6]), float(parts[7]),
            ])
    return (
        np.array(timestamps),
        np.array(positions),
        np.array(quats_wxyz),
    )


def pose_matrix_to_pos_quat_wxyz(c2w):
    """Convert 4x4 cam2world matrix to (position, quaternion_wxyz)."""
    pos = c2w[:3, 3]
    R = c2w[:3, :3]
    # scipy uses xyzw internally
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    # convert to wxyz for evo
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz


def evaluate_ate_evo(est_timestamps, est_positions, est_quats_wxyz,
                     gt_path):
    """Compute ATE using evo library with Sim(3) alignment."""
    from evo.core import sync
    from evo.core.metrics import PoseRelation
    from evo.core.trajectory import PoseTrajectory3D
    import evo.main_ape as main_ape

    traj_est = PoseTrajectory3D(
        positions_xyz=est_positions,
        orientations_quat_wxyz=est_quats_wxyz,
        timestamps=est_timestamps,
    )

    # Load GT — parse manually since the files are space-delimited EuRoC format
    gt_timestamps, gt_positions, gt_quats_wxyz = load_euroc_groundtruth(gt_path)
    traj_ref = PoseTrajectory3D(
        positions_xyz=gt_positions,
        orientations_quat_wxyz=gt_quats_wxyz,
        timestamps=gt_timestamps,
    )

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="flash-recon",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    return result


def build_cfg(args, intrinsics):
    """Build SLAM config for EuRoC evaluation."""
    fx, fy, cx, cy = intrinsics
    ht_out, wd_out = args.image_size

    return {
        "device": "cuda:0",
        "weights": {"droid": args.weights, "depth": args.depth_weights},
        "cam": {
            "H_out": ht_out,
            "W_out": wd_out,
            "down_scale": 8,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "tracking": {
            "beta": 0.3,
            "warmup": 15,
            "buffer": 512,
            "max_age": 20,
            "max_factors": 48,
            "mono_thres": 0.1,
            "depth_filter": {"thresh": 0.01, "n_views": 2},
            "local": {
                "nms": 1,
                "window": 20,
                "thresh": 17.5,
                "radius": 2,
                "max_factors": 48,
                "keyframe_thresh": 3.0,
                "enable_loop": False,
            },
            "loop_closure": {
                "window": 25,
                "thresh": 24.0,
                "radius": 2,
                "nms": 2,
            },
            "global": {
                "enabled": False,
                "freq": 20,
                "thresh": 24.0,
                "radius": 2,
                "nms": 2,
                "normalize": True,
            },
            "motion_filter": {
                "thresh": 2.4,
            },
        },
        "corrblock": {"num_levels": 4, "radius": 3, "max_slots": 128},
    }


def main():
    parser = argparse.ArgumentParser(description="EuRoC evaluation for flash-recon SLAM")
    parser.add_argument("--datapath", required=True, help="Path to EuRoC sequence (e.g. datasets/EuRoC/MH_01_easy)")
    parser.add_argument("--gt", default=None, help="Path to ground truth file (auto-resolved from datasets/EuRoC/groundtruth/)")
    parser.add_argument("--weights", default="neural/weights/droid.pth", help="Path to network weights")
    parser.add_argument("--depth_weights", default="neural/weights/depth_anything_v2_vits.pth", help="Path to DepthAnythingV2 weights")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride")
    parser.add_argument("--image_size", type=int, nargs=2, default=[320, 512], help="Output image size [H W]")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames to process (0 = all)")
    args = parser.parse_args()

    scene = os.path.basename(os.path.normpath(args.datapath))

    # Auto-resolve GT path
    if args.gt is None:
        euroc_root = os.path.dirname(os.path.normpath(args.datapath))
        # Handle grouped layout: datasets/EuRoC/machine_hall/MH_01_easy -> go up one more
        if os.path.basename(euroc_root) in ("machine_hall", "vicon_room1", "vicon_room2"):
            euroc_root = os.path.dirname(euroc_root)
        args.gt = os.path.join(euroc_root, "groundtruth", f"{scene}.txt")
    print(f"=== EuRoC evaluation: {scene} ===")
    print(f"  datapath: {args.datapath}")
    print(f"  gt: {args.gt}")
    print(f"  stride: {args.stride}")
    print(f"  image_size: {args.image_size}")

    # Build lazy image stream (no bulk loading into RAM)
    print("\nScanning images...")
    stream = EuRoCImageStream(
        args.datapath, args.image_size,
        stride=args.stride, max_frames=args.max_frames,
    )
    n_frames = len(stream)
    print(f"  {n_frames} frames (stride={args.stride})")

    # Build config and init models
    cfg = build_cfg(args, stream.intrinsics)
    device = cfg["device"]

    slam = SLAM(cfg)
    print("  SLAM initialized")

    # Run SLAM
    print(f"\nProcessing {n_frames} frames...")
    times = []

    try:
        for i in range(n_frames):
            tstamp_ns, frame = stream[i]
            frame = frame.to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            slam(frame, tstamp=tstamp_ns)

            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)

            if i % 100 == 0:
                n_edges = len(slam)
                print(
                    f"  frame {i:4d}/{n_frames} | kf: {len(slam.buffer)} | "
                    f"edges: {n_edges} | {dt*1000:.0f}ms"
                )
    except torch.OutOfMemoryError:
        print(f"\n  OOM at frame {i}")

    n_kf = len(slam.buffer)
    print(f"\nProcessed {len(times)} frames, {n_kf} keyframes")
    if times:
        print(
            f"Timing: avg {np.mean(times)*1000:.0f}ms, "
            f"median {np.median(times)*1000:.0f}ms, "
            f"p80 {np.percentile(times, 80)*1000:.0f}ms"
        )

    # Terminal global BA (mirrors DROID-SLAM's backend(7) + backend(12))
    print("\nRunning terminal global BA...")
    torch.cuda.empty_cache()
    t_final = time.perf_counter()
    slam.finalize(steps_per_pass=(7, 12))
    print(f"  Terminal BA: {time.perf_counter() - t_final:.1f}s")

    # Extract estimated trajectory from keyframe poses
    est_timestamps = []
    est_positions = []
    est_quats_wxyz = []

    for kf_idx in range(n_kf):
        tstamp_ns = slam.buffer.get_tstamp(kf_idx)
        c2w = slam.buffer.get_cam2world(kf_idx).cpu().numpy()
        pos, quat = pose_matrix_to_pos_quat_wxyz(c2w)

        est_timestamps.append(tstamp_ns / 1e9)  # seconds for evo
        est_positions.append(pos)
        est_quats_wxyz.append(quat)

    est_timestamps = np.array(est_timestamps)
    est_positions = np.array(est_positions)
    est_quats_wxyz = np.array(est_quats_wxyz)

    print(f"\nTrajectory: {len(est_positions)} keyframes")
    if len(est_positions) < 3:
        print("Not enough keyframes — cannot compute ATE")
        return

    est_travel = np.linalg.norm(np.diff(est_positions, axis=0), axis=1).sum()
    print(f"  Est travel: {est_travel:.3f}")

    # Evaluate ATE with evo
    print("\nComputing ATE (evo, Sim(3) alignment)...")
    result = evaluate_ate_evo(est_timestamps, est_positions, est_quats_wxyz, args.gt)

    ate_rmse = result.stats["rmse"]
    ate_mean = result.stats["mean"]
    ate_median = result.stats["median"]
    ate_max = result.stats["max"]

    print(f"\n  ATE RMSE:   {ate_rmse:.4f} m")
    print(f"  ATE mean:   {ate_mean:.4f} m")
    print(f"  ATE median: {ate_median:.4f} m")
    print(f"  ATE max:    {ate_max:.4f} m")
    print(f"  Keyframes:  {len(est_positions)}")

    # Print single-line summary for batch parsing
    print(f"\nRESULT {scene} {ate_rmse:.6f}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
