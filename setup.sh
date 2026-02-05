#!/bin/bash

uv sync --all-extras --group dev --preview-features extra-build-dependencies

.venv/bin/python -c "
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


def test_module(module_name):
    try:
        mod = __import__(module_name)
    except ImportError as e:
        print(f\"{module_name}: {e}\")
        return False
    print(f\"{module_name}: ok\")
    return True


def test_camera():
    try:
        camera = Camera(\"/dev/mapping-cam\", deterministic=True)
        frame = next(camera)
    except Exception as e:
        print(f\"camera: {e}\")
        return False
    print(\"camera: ok\")
    return True


def test_gpu():
    try:
        import torch

        frame = torch.ones(240, 320, 2, device=\"cuda\", dtype=torch.uint8)
        out = model(frame)
    except Exception as e:
        print(f\"gpu: {e}\")
        return False
    print(\"gpu: ok\")
    return True

print()
succeeded = True

succeeded &= test_module(\"cv2\")
succeeded &= test_module(\"torch\")
succeeded &= test_module(\"torch_scatter\")
succeeded &= test_module(\"lietorch\")
succeeded &= test_module(\"visionrt\")
succeeded &= test_module(\"geometry\")
succeeded &= test_module(\"neural\")
succeeded &= test_module(\"flash_recon\")
succeeded &= test_camera()
succeeded &= test_gpu()

print()
if succeeded:
    print(\"Environment setup completed successfully\")
else:
    print(\"Environment setup failed\")
"
