#!/bin/bash

uv sync --all-extras --group dev --preview-features extra-build-dependencies

.venv/bin/python -c "
def test(module_name):
    try:
        mod = __import__(module_name)
    except ImportError as e:
        print(f\"{module_name}: {e}\")
        return False
    print(f\"{module_name}: ok\")
    return True

print()
test(\"torch\")
test(\"cv2\")

test(\"lietorch\")
test(\"torch_scatter\")

test(\"visionrt\")
test(\"geometry\")
test(\"neural\")
test(\"flash_recon\")

print()

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


from visionrt import config, compile, Camera, Preprocessor


config.verbose = False
config.optims.fold_conv_bn = True
config.cudagraphs = True


model = compile(
    nn.Sequential(
        Preprocessor(use_triton=False),
        nn.Upsample(size=(224, 224), mode=\"bilinear\", align_corners=False),
        resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        nn.Softmax(dim=1),
    )
    .cuda()
    .eval()
)

try:
    camera = Camera(\"/dev/mapping-cam\", deterministic=True)
    frame = next(camera)
except Exception as e:
    print(\"Camera test failed: {e}\")
    exit()

print(\"Camera test succeeded\")

try:
    out = model(frame)
except Exception as e:
    print(f\"GPU test failed: {e}\")
    exit()

print(\"GPU test succeeded\")

print(\"Environment setup completed successfully\")
"
