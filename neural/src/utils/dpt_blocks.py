import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, x, res=None, size=None):
        if res is not None:
            x = x + self.resConfUnit1(res)
        x = self.resConfUnit2(x)
        x = self.out_conv(x)
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return x


def make_scratch(in_shape, out_shape):
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.refinenet1 = FeatureFusionBlock(out_shape)
    scratch.refinenet2 = FeatureFusionBlock(out_shape)
    scratch.refinenet3 = FeatureFusionBlock(out_shape)
    scratch.refinenet4 = FeatureFusionBlock(out_shape)
    scratch.output_conv1 = nn.Conv2d(
        out_shape, out_shape // 2, kernel_size=3, stride=1, padding=1
    )
    scratch.output_conv2 = nn.Sequential(
        nn.Conv2d(out_shape // 2, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True),
        nn.Identity(),
    )
    return scratch
