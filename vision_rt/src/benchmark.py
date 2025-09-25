"""
VisionRT vs. Standard (OpenCV + PyTorch) Benchmark

This script reproduces the performance comparison shown in the README.
It measures end-to-end throughput of a simple ResNet50 classification pipeline.

Note: This benchmark sets OpenCV's CAP_PROP_BUFFERSIZE=1 to force single-buffer
capture for fair comparison with the current VisionRT implementation.
Future versions of VisionRT will support multi-buffer asynchronous capture,
at which point this limitation will be removed.
"""

import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np

import vision_rt_cuda as vrt


def benchmark_standard(num_frames=100):
    print("=== Standard Pipeline Benchmark ===")

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        def forward(self, x: torch.Tensor):
            if x.ndim == 3:
                x = x.unsqueeze(0)
            x = x / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            return self.model(x)

    model = Classifier().cuda().eval()

    cap = cv2.VideoCapture("/dev/mapping-cam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 90)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start = time.time()
    frames = 0

    while frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1))
        tensor = torch.from_numpy(chw).cuda().float()

        with torch.no_grad():
            _ = model(tensor)
            torch.cuda.synchronize()

        frames += 1

    cap.release()
    elapsed = time.time() - start
    fps = frames / elapsed
    print(f"Standard FPS: {fps:.1f}")
    return fps


def benchmark_visionrt(num_frames=100):
    print("\n=== VisionRT Benchmark ===")

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        def forward(self, x: torch.Tensor):
            if x.ndim == 3:
                x = x.unsqueeze(0)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            return self.model(x)

    camera = vrt.Camera("/dev/mapping-cam")
    model = Classifier().cuda().eval()
    rt_model = vrt.GraphCachedModule(model)

    # Capture CUDA graph & compile
    example = torch.randn(1, 3, 240, 320).cuda()
    rt_model.capture(example)

    start = time.time()
    frames = 0

    for frame in camera.stream():
        with torch.no_grad():
            _ = rt_model(frame)
            torch.cuda.synchronize()
        frames += 1
        if frames >= num_frames:
            break

    camera.close()
    elapsed = time.time() - start
    fps = frames / elapsed
    print(f"VisionRT FPS: {fps:.1f}")
    return fps


if __name__ == "__main__":
    print("VisionRT vs. Standard Benchmark")
    print("=" * 50)

    standard_fps = benchmark_standard(100)
    visionrt_fps = benchmark_visionrt(100)

    print("\n=== Summary ===")
    print(f"Standard FPS: {standard_fps:.1f}")
    print(f"VisionRT FPS: {visionrt_fps:.1f}")
    print(f"Speedup: {visionrt_fps / standard_fps:.1f}×")
