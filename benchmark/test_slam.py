import os
import time
import numpy as np
import torch
from PIL import Image

from flash_recon.slam import SLAM

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
MAX_FRAMES = 50

cfg = {
    "device": "cuda:0",
    "weights": {"droid": WEIGHTS_PATH},
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
        "buffer": 512,
        "max_age": 25,
        "max_factors": 75,
        "depth_filter": {"thresh": 0.01, "n_views": 2},
        "local": {
            "window": 25,
            "radius": 2,
            "nms": 1,
            "thresh": 16.0,
            "max_factors": 75,
        },
        "global": {
            "window": 25,
            "radius": 1,
            "nms": 5,
            "thresh": 25.0,
        },
        "loop": {
            "window": 25,
            "radius": 1,
            "nms": 12,
            "thresh": 25.0,
        },
    },
    "global_ba": {"freq": 20},
    "motion_threshold": 4.0,
    "dist_threshold": 4.0,
    "publish_freq": 1,
    "corrblock": {"num_levels": 4, "radius": 3},
    "profile": False,
}


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


def load_and_preprocess(path, device):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if H_EDGE > 0 or W_EDGE > 0:
        img = img[H_EDGE:-H_EDGE, W_EDGE:-W_EDGE]
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.resize((W_OUT, H_OUT), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def make_dummy_mono_depth(device):
    return torch.ones(H_OUT // 8, W_OUT // 8, device=device)


def main():
    device = cfg["device"]
    rgb_list = load_tum_rgb_list(DATASET_ROOT)
    print(f"Dataset: {DATASET_ROOT} ({len(rgb_list)} frames, using {MAX_FRAMES})")

    slam = SLAM(cfg)
    print("SLAM initialized")

    times = []
    try:
        for i, (ts, path) in enumerate(rgb_list[:MAX_FRAMES]):
            frame = load_and_preprocess(path, device)
            mono_depth = make_dummy_mono_depth(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            slam(frame, mono_depth)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)

            if i % 10 == 0:
                print(f"  frame {i:4d} | keyframes: {len(slam.buffer)} | {dt*1000:.0f}ms")
    except torch.OutOfMemoryError:
        print(f"\n  OOM at frame {i} — printing results so far")

    n_kf = len(slam.buffer)
    avg_ms = np.mean(times) * 1000
    med_ms = np.median(times) * 1000
    print(f"\nDone. Keyframes: {n_kf}, frames processed: {len(times)}")
    print(f"Timing: avg {avg_ms:.0f}ms, median {med_ms:.0f}ms per frame")
    print(f"\nVO breakdown:")
    print(slam.vo.timer.summary())

    # trajectory sanity check
    translations = []
    for idx in range(n_kf):
        c2w = slam.buffer.get_cam2world(idx).cpu().numpy()
        translations.append(c2w[:3, 3])
    translations = np.array(translations)
    travel = np.linalg.norm(np.diff(translations, axis=0), axis=1).sum()
    span = np.linalg.norm(translations.max(axis=0) - translations.min(axis=0))
    has_nan = np.any(np.isnan(translations))
    print(f"\nTrajectory: {n_kf} keyframes, travel={travel:.3f}m, span={span:.3f}m, nan={has_nan}")
    if has_nan or travel < 1e-6 or span < 1e-6:
        print("FAILED — degenerate trajectory")
    else:
        print("PASSED" if n_kf > 1 else "FAILED")


# python3 -m benchmark.slam_test
if __name__ == "__main__":
    print("=== Run 1 (includes compile warmup) ===")
    main()
    print("\n=== Run 2 (compiled, warm) ===")
    main()
