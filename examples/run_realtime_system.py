import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import torch
from PIL import Image

from flash_recon import System

H_OUT, W_OUT = 240, 320
FX = 128.0
FY = 128.0
CX = 160.0
CY = 120.0

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


def load_rgb(path, device):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.resize((W_OUT, H_OUT), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).to(device)


def preprocess(img, device):
    frame = img.permute(2, 0, 1).unsqueeze(0)
    return (frame - MEAN.to(device)) / STD.to(device)

from visionrt import Camera, Preprocessor

ppc = Preprocessor()
def main():
    #device = cfg["device"]
    camera = Camera("/dev/mapping-cam")
    system = System(cfg, persist=True)

    def _f():
        dev = torch.device(cfg["device"])
        mean = MEAN.to(dev)
        std = STD.to(dev)
        for yuyv in camera.stream():
            frame = ppc(yuyv)                        # [1, 3, H, W] normalized
            raw = frame.squeeze(0) * std + mean      # denormalize -> [3, H, W] in [0,1]
            raw = raw.permute(1, 2, 0).clamp(0, 1)   # -> [H, W, 3]
            yield frame, raw
        
    system.set_frame_generator(_f())

    system.run()


if __name__ == "__main__":
    main()
