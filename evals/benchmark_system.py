import argparse
import warnings
warnings.filterwarnings("ignore")

import os
import glob
import time
import numpy as np
import torch
from PIL import Image

from flash_recon.system import System
from evals.eval_euroc import EuRoCImageStream

H_OUT, W_OUT = 320, 512

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DATASETS = {
    "tum": {
        "root": "datasets/TUM/fr1_desk",
        "H_edge": 8,
        "W_edge": 8,
        "fx_native": 517.3,
        "fy_native": 516.5,
        "cx_native": 318.6,
        "cy_native": 255.3,
    },
    "tartanair": {
        "root": "datasets/tartanair_test",
        # 640x480, intrinsics scaled by 0.8 to match 512x384 output
        "fx": 320.0 * 0.8,
        "fy": 320.0 * 0.8,
        "cx": 320.0 * 0.8,
        "cy": 240.0 * 0.8,
    },
    "euroc": {
        "root": "datasets/EuRoC",
        "image_size": [320, 512],
        "stride": 1,
    },
}


def get_cam_cfg(dataset, datapath=None, euroc_stream=None):
    d = DATASETS[dataset]

    if dataset == "tum":
        he, we = d["H_edge"], d["W_edge"]
        h_crop = 480 - 2 * he
        w_crop = 640 - 2 * we
        fx = d["fx_native"] * W_OUT / w_crop
        fy = d["fy_native"] * H_OUT / h_crop
        cx = (d["cx_native"] - we) * W_OUT / w_crop
        cy = (d["cy_native"] - he) * H_OUT / h_crop
    elif dataset == "euroc":
        fx, fy, cx, cy = euroc_stream.intrinsics
    else:
        fx, fy, cx, cy = d["fx"], d["fy"], d["cx"], d["cy"]

    h_out = d.get("image_size", [H_OUT, W_OUT])[0] if dataset == "euroc" else H_OUT
    w_out = d.get("image_size", [H_OUT, W_OUT])[1] if dataset == "euroc" else W_OUT

    return {"H_out": h_out, "W_out": w_out, "down_scale": 8,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy}


def load_tum_images(root):
    rgb_txt = os.path.join(root, "rgb.txt")
    paths = []
    with open(rgb_txt) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            _, path = line.split()
            paths.append(os.path.join(root, path))
    paths.sort()
    return paths


def load_tartanair_images(root, sequence=None):
    if sequence:
        img_dir = os.path.join(root, sequence, "image_left")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(root, sequence)
    else:
        img_dir = os.path.join(root, "image_left")
        if not os.path.isdir(img_dir):
            img_dir = root
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    return paths


def load_rgb(path, device, dataset="tum"):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if dataset == "tum":
        he, we = DATASETS["tum"]["H_edge"], DATASETS["tum"]["W_edge"]
        if he > 0 or we > 0:
            img = img[he:-he, we:-we]
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.resize((W_OUT, H_OUT), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).to(device)


def preprocess(img, device):
    frame = img.permute(2, 0, 1).unsqueeze(0)
    return (frame - MEAN.to(device)) / STD.to(device)


class BenchmarkedSystem(System):
    def __init__(self, cfg, persist=False):
        super().__init__(cfg, persist)
        self.slam_times = []
        self.splat_times = []

    def _slammer(self, frame, raw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        super()._slammer(frame, raw)
        torch.cuda.synchronize()
        self.slam_times.append(time.perf_counter() - t0)

    def _splatter(self):
        while self._running:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            from queue import Empty
            try:
                keyframe = self._keyframe_q.get_nowait()
                if self.splat.add(keyframe):
                    pass
            except Empty:
                pass

            snapshot = self.slam.snapshot
            loss = self.splat(snapshot)

            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if loss != 0.0:
                self.splat_times.append(dt)
                if not self._slam_done:
                    time.sleep(1 / 30)
            else:
                time.sleep(1 / 30)

    def report(self):
        def stats(times, name):
            t = np.array(times) * 1000
            print(f"\n  {name}:")
            print(f"frames: {len(t)}")
            print(f"median: {np.median(t):.1f} ms")
            print(f"mean: {np.mean(t):.1f} ms")
            print(f"p95: {np.percentile(t, 95):.1f} ms")
            print(f"max: {np.max(t):.1f} ms")
            print(f"fps: {1000 / np.median(t):.1f}")

        print("\n========================================")
        print("  System Benchmark")
        print("========================================")
        if self.slam_times:
            stats(self.slam_times, "slammer")
        if self.splat_times:
            stats(self.splat_times, "splatter")


WEIGHTS_PATH = os.environ.get("DROID_WEIGHTS", "neural/weights/droid.pth")

TRACKING_CFG = {
    "beta": 0.75,
    "warmup": 12,
    "buffer": 256,
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
        "keyframe_thresh": 3.0,
        "enable_loop": False,
    },
    "loop_closure": {
        "window": 25,
        "thresh": 25.0,
        "radius": 1,
        "nms": 10,
    },
    "global": {
        "freq": 20,
        "thresh": 25.0,
        "radius": 1,
        "nms": 5,
        "normalize": False,
        "enabled": False,
    },
    "motion_filter": {"thresh": 4},
}

SPLATTING_CFG = {
    "stride": 3,
    "n_views": 16,
    "lr": {
        "means": 1e-4,
        "colors": 1e-3,
        "scales": 5e-3,
        "quats": 1e-3,
        "alphas": 5e-2,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tum", "tartanair", "euroc"], default="tum")
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--sequence", type=str, default=None,
                        help="TartanAir sequence name, e.g. ME000")
    args = parser.parse_args()

    dataset = args.dataset
    datapath = args.datapath or DATASETS[dataset]["root"]
    device = "cuda:0"

    euroc_stream = None
    if dataset == "euroc":
        d = DATASETS["euroc"]
        euroc_stream = EuRoCImageStream(
            datapath, d["image_size"], stride=d["stride"],
        )

    cam_cfg = get_cam_cfg(dataset, datapath, euroc_stream=euroc_stream)

    cfg = {
        "device": device,
        "weights": {"droid": WEIGHTS_PATH},
        "cam": cam_cfg,
        "tracking": TRACKING_CFG,
        "corrblock": {"num_levels": 4, "radius": 3, "max_slots": 128},
        "splatting": SPLATTING_CFG,
    }

    if dataset == "euroc":
        n_frames = len(euroc_stream)
        print(f"Dataset: {dataset} | {n_frames} frames (stride={DATASETS['euroc']['stride']}) | {datapath}")

        system = BenchmarkedSystem(cfg, persist=False)

        def _f():
            for i in range(n_frames):
                _, frame = euroc_stream[i]
                frame = frame.to(device)
                # euroc_stream returns preprocessed frames, compute raw rgb for splatter
                rgb_tensor = (frame.squeeze(0) * STD.to(device) + MEAN.to(device)).clamp(0, 1)
                rgb = rgb_tensor.permute(1, 2, 0)
                yield frame, rgb

        system.set_frame_generator(_f())
    else:
        if dataset == "tum":
            paths = load_tum_images(datapath)
        else:
            paths = load_tartanair_images(datapath, args.sequence)

        print(f"Dataset: {dataset} | {len(paths)} frames | {datapath}")

        system = BenchmarkedSystem(cfg, persist=False)

        def _f():
            for path in paths:
                rgb = load_rgb(path, device, dataset)
                frame = preprocess(rgb, device)
                yield frame, rgb

        system.set_frame_generator(_f())

    system.run()
    system.report()


if __name__ == "__main__":
    main()
