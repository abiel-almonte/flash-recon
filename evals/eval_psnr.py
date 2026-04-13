import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import gsplat
from PIL import Image

from flash_recon.system import System

H_OUT, W_OUT = 384, 512
H_EDGE, W_EDGE = 8, 8

TUM_INTRINSICS = {
    "fr1": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3},
    "fr2": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7},
    "fr3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6},
}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
WEIGHTS_PATH = os.environ.get("DROID_WEIGHTS", "neural/weights/droid.pth")


def get_intrinsics(datapath):
    basename = os.path.basename(os.path.normpath(datapath))
    for prefix in ("fr3", "fr2", "fr1"):
        if basename.startswith(prefix):
            return TUM_INTRINSICS[prefix]
    return TUM_INTRINSICS["fr1"]


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


def load_rgb(path, device):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if H_EDGE > 0 or W_EDGE > 0:
        img = img[H_EDGE:-H_EDGE, W_EDGE:-W_EDGE]
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.resize((W_OUT, H_OUT), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).to(device)


def preprocess(img, device):
    frame = img.permute(2, 0, 1).unsqueeze(0)
    return (frame - MEAN.to(device)) / STD.to(device)


def compute_psnr(rendered, gt):
    mse = (rendered - gt).pow(2).mean()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(1.0 / mse).item()


def compute_ssim(img1, img2, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    x = img1.permute(2, 0, 1).unsqueeze(0)
    y = img2.permute(2, 0, 1).unsqueeze(0)
    B, Ch = x.shape[:2]
    x = x.reshape(B * Ch, 1, x.shape[2], x.shape[3])
    y = y.reshape(B * Ch, 1, y.shape[2], y.shape[3])
    kernel = torch.ones(1, 1, window_size, window_size, device=x.device) / (window_size ** 2)
    mu_x = torch.nn.functional.conv2d(x, kernel, padding=window_size // 2)
    mu_y = torch.nn.functional.conv2d(y, kernel, padding=window_size // 2)
    sigma_x_sq = torch.nn.functional.conv2d(x * x, kernel, padding=window_size // 2) - mu_x ** 2
    sigma_y_sq = torch.nn.functional.conv2d(y * y, kernel, padding=window_size // 2) - mu_y ** 2
    sigma_xy = torch.nn.functional.conv2d(x * y, kernel, padding=window_size // 2) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="datasets/TUM/fr3_office")
    args = parser.parse_args()

    intr = get_intrinsics(args.datapath)
    H_CROP = 480 - 2 * H_EDGE
    W_CROP = 640 - 2 * W_EDGE
    fx = intr["fx"] * W_OUT / W_CROP
    fy = intr["fy"] * H_OUT / H_CROP
    cx = (intr["cx"] - W_EDGE) * W_OUT / W_CROP
    cy = (intr["cy"] - H_EDGE) * H_OUT / H_CROP

    device = "cuda:0"

    cfg = {
        "device": device,
        "weights": {"droid": WEIGHTS_PATH},
        "cam": {
            "H_out": H_OUT, "W_out": W_OUT, "down_scale": 8,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
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
        "splatting": {
            "stride": 3,
            "n_views": 16,
            "lr": {
                "means": 1e-4, "colors": 1e-3, "scales": 5e-3,
                "quats": 1e-3, "alphas": 5e-2,
            },
        },
    }

    rgb_list = load_tum_rgb_list(args.datapath)
    print(f"Dataset: {args.datapath} | {len(rgb_list)} frames")

    import time as _time
    from threading import Thread

    settle_time = 30

    system = System(cfg, persist=True)

    def _f():
        for _, path in rgb_list:
            rgb = load_rgb(path, device)
            frame = preprocess(rgb, device)
            yield frame, rgb

    system.set_frame_generator(_f())

    run_thread = Thread(target=system.run)
    run_thread.start()

    while not system._slam_done:
        _time.sleep(0.5)

    print(f"\nSLAM done. Letting splatter optimize for {settle_time}s...")
    _time.sleep(settle_time)

    # Stop the system
    system._running = False
    run_thread.join()

    splat_buf = system.splat.buffer
    K = system.splat.optimizer.K.to(device)

    n_kf = splat_buf._n_keyframes
    if n_kf == 0:
        print("No keyframes — cannot evaluate")
        return

    colors = splat_buf.colors.clamp(0, 1)
    scales = splat_buf.scales.exp()
    opacities = splat_buf.alphas.sigmoid()
    quats = splat_buf.quats / splat_buf.quats.norm(dim=-1, keepdim=True)

    psnrs = []
    ssims = []

    with torch.no_grad():
        for i in range(n_kf):
            viewmat = splat_buf.viewmats[i:i+1]
            gt = splat_buf.gts[i]  # [H, W, 3]

            rendered, _, _ = gsplat.rasterization(
                means=splat_buf.means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=K,
                width=W_OUT,
                height=H_OUT,
            )

            rendered = rendered.squeeze(0).clamp(0, 1)  # [H, W, 3]
            psnr = compute_psnr(rendered, gt)
            ssim = compute_ssim(rendered, gt)
            psnrs.append(psnr)
            ssims.append(ssim)

    psnrs = np.array(psnrs)
    ssims = np.array(ssims)

    print(f"\n========================================")
    print(f"  Rendering Quality ({n_kf} keyframes)")
    print(f"========================================")
    print(f"  PSNR:  {np.mean(psnrs):.2f} dB (median: {np.median(psnrs):.2f})")
    print(f"  SSIM:  {np.mean(ssims):.4f} (median: {np.median(ssims):.4f})")


if __name__ == "__main__":
    main()
