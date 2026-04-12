import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import torch
from PIL import Image

from flash_recon import System

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

DATASET_ROOT = os.environ.get("DATASET_ROOT", "datasets/TUM/fr1_desk")
WEIGHTS_PATH = os.environ.get("DROID_WEIGHTS", "neural/weights/droid.pth")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

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
            "keyframe_thresh": 3.0,
            "enable_loop": True,
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
        "motion_filter": {
            "thresh": 4,
        },
    },
    "corrblock": {"num_levels": 4, "radius": 3, "max_slots": 128},
    "splatting": {
        "stride": 3,
        "n_views": 16,
        "lr": {
            "means": 1e-4,
            "colors": 1e-3,
            "scales": 5e-3,
            "quats": 1e-3,
            "alphas": 5e-2,
        },
    },
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


def main():
    device = cfg["device"]
    rgb_list = load_tum_rgb_list(DATASET_ROOT)

    system = System(cfg, persist=True)

    def _f():
        for _, path in rgb_list:

            rgb = load_rgb(path, device)
            frame = preprocess(rgb, device)

            yield frame, rgb
        
    system.set_frame_generator(_f())

    system.run()


if __name__ == "__main__":
    main()
