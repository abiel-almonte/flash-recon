"""Reference DROID-SLAM evaluation on EuRoC MAV dataset."""

import sys
import os
import glob
import argparse
import time

import cv2
import numpy as np
import torch

# DROID-SLAM must be installed separately (pip install -e <path-to-DROID-SLAM>)
# If using a local checkout, set DROID_SLAM_DIR env var to point to droid_slam/
DROID_SLAM_DIR = os.environ.get("DROID_SLAM_DIR")
if DROID_SLAM_DIR:
    sys.path.insert(0, DROID_SLAM_DIR)

from pathlib import Path
from tqdm import tqdm


def image_stream(datapath, image_size=[320, 512], stereo=False, stride=1):
    """EuRoC image generator — verbatim from DROID-SLAM test_euroc.py."""

    K_l = np.array([458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0]).reshape(3, 3)
    d_l = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
    R_l = np.array([
        0.999966347530033, -0.001422739138722922, 0.008079580483432283,
        0.001365741834644127, 0.9999741760894847, 0.007055629199258132,
        -0.008089410156878961, -0.007044357138835809, 0.9999424675829176
    ]).reshape(3, 3)

    P_l = np.array([435.2046959714599, 0, 367.4517211914062, 0, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1, 0]).reshape(3, 4)
    map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3, :3], (752, 480), cv2.CV_32F)

    K_r = np.array([457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1]).reshape(3, 3)
    d_r = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]).reshape(5)
    R_r = np.array([
        0.9999633526194376, -0.003625811871560086, 0.007755443660172947,
        0.003680398547259526, 0.9999684752771629, -0.007035845251224894,
        -0.007729688520722713, 0.007064130529506649, 0.999945173484644
    ]).reshape(3, 3)

    P_r = np.array([435.2046959714599, 0, 367.4517211914062, -47.90639384423901, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1, 0]).reshape(3, 4)
    map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3, :3], (752, 480), cv2.CV_32F)

    intrinsics_vec = [435.2046959714599, 435.2046959714599, 367.4517211914062, 252.2008514404297]
    ht0, wd0 = [480, 752]

    # Resolve grouped EuRoC archive layout
    cam0_dir = os.path.join(datapath, 'mav0/cam0/data')
    if not os.path.isdir(cam0_dir):
        scene = os.path.basename(os.path.normpath(datapath))
        parent = os.path.dirname(os.path.normpath(datapath))
        if scene.startswith("MH"):
            datapath = os.path.join(parent, "machine_hall", scene)
        elif scene.startswith("V1"):
            datapath = os.path.join(parent, "vicon_room1", scene)
        elif scene.startswith("V2"):
            datapath = os.path.join(parent, "vicon_room2", scene)

    images_left = sorted(glob.glob(os.path.join(datapath, 'mav0/cam0/data/*.png')))[::stride]
    images_right = [x.replace('cam0', 'cam1') for x in images_left]

    data_list = []
    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        if stereo and not os.path.isfile(imgR):
            continue
        tstamp = float(imgL.split('/')[-1][:-4])
        images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        if stereo:
            images += [cv2.remap(cv2.imread(imgR), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]

        images = [cv2.resize(image, (image_size[1], image_size[0])) for image in images]
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to(dtype=torch.float32)

        intrinsics = torch.as_tensor(intrinsics_vec)
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        data_list.append((stride * t, images, intrinsics))

    return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reference DROID-SLAM evaluation on EuRoC")
    parser.add_argument("--datapath", required=True, help="Path to EuRoC sequence")
    parser.add_argument("--gt", required=True, help="Path to ground truth file (e.g. datasets/EuRoC/groundtruth/MH_01_easy.txt)")
    parser.add_argument("--weights", default="neural/weights/droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[320, 512])
    parser.add_argument("--stereo", action="store_true")

    # DROID-SLAM default EuRoC parameters
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--keyframe_thresh", type=float, default=3.0)
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=24.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)

    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    args = parser.parse_args()

    # Required by Droid but not in argparse — set to match test_euroc.py
    args.disable_vis = True

    torch.multiprocessing.set_start_method('spawn')

    scene = Path(args.datapath).name
    print(f"=== DROID-SLAM reference evaluation: {scene} ===")
    print(f"  datapath: {args.datapath}")
    print(f"  gt: {args.gt}")
    print(f"  weights: {args.weights}")
    print(args)

    # Import Droid after sys.path setup
    from droid import Droid

    droid = Droid(args)

    # Load images with stride 2 (for tracking)
    print("\nLoading images...")
    images = image_stream(args.datapath, stereo=args.stereo, stride=2)
    print(f"  {len(images)} frames (stride 2)")

    # Track
    t0 = time.perf_counter()
    for (t, image, intrinsics) in tqdm(images, desc=scene):
        droid.track(t, image, intrinsics=intrinsics)
    tracking_time = time.perf_counter() - t0

    # Backend global BA (no trajectory filling)
    print("\nRunning backend optimization...")
    del droid.frontend
    torch.cuda.empty_cache()
    droid.backend(7)
    torch.cuda.empty_cache()
    droid.backend(12)

    total_time = time.perf_counter() - t0
    print(f"\nTiming: tracking {tracking_time:.1f}s, total {total_time:.1f}s")

    # Extract keyframe poses
    from lietorch import SE3

    N = droid.video.counter.value
    print(f"  {N} keyframes")

    poses_w2c = SE3(droid.video.poses[:N])
    poses_c2w = poses_w2c.inv()
    traj_data = poses_c2w.data.cpu().numpy()  # [N, 7]: tx ty tz qx qy qz qw

    kf_tstamps = droid.video.tstamp[:N].cpu().numpy()

    # Convert lietorch quat (qx,qy,qz,qw) -> evo quat (qw,qx,qy,qz)
    positions = traj_data[:, :3]
    quats_xyzw = traj_data[:, 3:]
    quats_wxyz = np.column_stack([quats_xyzw[:, 3], quats_xyzw[:, :3]])

    ### Evaluation — keyframe-only, Sim(3) alignment ###
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    # Map keyframe indices back to nanosecond timestamps
    images_list = sorted(glob.glob(os.path.join(args.datapath, 'mav0/cam0/data/*.png')))
    if not images_list:
        resolved = args.datapath
        sname = os.path.basename(os.path.normpath(resolved))
        parent = os.path.dirname(os.path.normpath(resolved))
        if sname.startswith("MH"):
            resolved = os.path.join(parent, "machine_hall", sname)
        elif sname.startswith("V1"):
            resolved = os.path.join(parent, "vicon_room1", sname)
        elif sname.startswith("V2"):
            resolved = os.path.join(parent, "vicon_room2", sname)
        images_list = sorted(glob.glob(os.path.join(resolved, 'mav0/cam0/data/*.png')))

    all_tstamps_ns = np.array([float(x.split('/')[-1][:-4]) for x in images_list])

    # DROID stores frame indices as tstamps (stride-2 indices: 0, 2, 4, ...)
    # Map back to actual nanosecond timestamps
    kf_tstamps_ns = []
    for idx in kf_tstamps:
        frame_idx = int(round(idx))
        if 0 <= frame_idx < len(all_tstamps_ns):
            kf_tstamps_ns.append(all_tstamps_ns[frame_idx])
        else:
            kf_tstamps_ns.append(idx)  # fallback
    kf_tstamps_ns = np.array(kf_tstamps_ns)

    traj_est = PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=quats_wxyz,
        timestamps=kf_tstamps_ns)

    # Load GT
    from evo.tools import file_interface
    traj_ref = file_interface.read_tum_trajectory_file(args.gt)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result = main_ape.ape(traj_ref, traj_est, est_name='droid-slam',
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)

    ate_rmse = result.stats["rmse"]
    print(f"  Keyframes: {N}")
    print(f"\nRESULT {scene} {ate_rmse:.6f}")
