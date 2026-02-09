import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import DinoVisionTransformer, make_scratch


class DPTHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        in_channels = 384
        features = 64
        out_channels = [48, 96, 192, 384]

        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels, oc, kernel_size=1) for oc in out_channels]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2
                ),
                nn.Identity(),
                nn.Conv2d(
                    out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = make_scratch(out_channels, features)

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x[0]  # patch tokens (discard class token)
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=True
        )
        out = self.scratch.output_conv2(out)
        return out

# DepthAnythingV2
class MonoDepth(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.pretrained = DinoVisionTransformer()
        self.depth_head = DPTHead()
        self._load_weights(cfg["weights"]["depth"])

        H_out = int(cfg.get("cam", {}).get("H_out", 480))
        W_out = int(cfg.get("cam", {}).get("W_out", 640))
        down_scale = int(cfg.get("cam", {}).get("down_scale", 8))

        self.ht = H_out // down_scale
        self.wd = W_out // down_scale

    def _load_weights(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(frame, size=(518, 518), mode="bilinear", align_corners=False)
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x, [2, 5, 8, 11], return_class_token=True
        )
        depth = F.relu(self.depth_head(features, patch_h, patch_w))
        depth = depth.clamp(min=1e-3)
        depth = F.interpolate(depth, size=(self.ht, self.wd), mode="bilinear", align_corners=False)
        return depth.squeeze(0).squeeze(0)
