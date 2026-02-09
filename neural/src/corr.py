import torch
import torch.nn.functional as F

from neural_cuda.corr import corr_forward, altcorr_forward


class CorrBlock:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.num_levels = int(cfg.get("corrblock", {}).get("num_levels", 4))
        self.radius = int(cfg.get("corrblock", {}).get("radius", 3))
        self.pyramid = None

    def build_pyramid(
        self, feature1: torch.Tensor, feature2: torch.Tensor
    ):  # features will be [T, c, h, w] each
        T, c, ht, wd = feature1.shape
        p = ht * wd

        f1 = feature1.reshape(T, c, ht * wd) / 4.0
        f2 = feature2.reshape(T, c, ht * wd) / 4.0

        corr = torch.bmm(f1.transpose(-2, -1), f2)  # [T, p, p]
        corr = corr.view(T * p, 1, ht, wd)

        pyramid = []
        for i in range(self.num_levels):
            Hi = corr.shape[-2]
            Wi = corr.shape[-1]

            level = corr.view(T, ht, wd, Hi, Wi).half()

            if self.pyramid:
                pyramid.append(torch.cat([self.pyramid[i], level], dim=0))
            else:
                pyramid.append(level)

            if i + 1 < self.num_levels:
                corr = F.avg_pool2d(corr, kernel_size=2, stride=2)

        self.pyramid = pyramid

    def clear_pyramid(self):
        self.pyramid = None

    def filter_pyramid(self, keep: torch.Tensor):
        if self.pyramid is None:
            return
        self.pyramid = [level[keep] for level in self.pyramid]

    def __getitem__(self, keep: torch.Tensor) -> "CorrBlock":
        new_corr = CorrBlock(self.cfg)
        if self.pyramid is not None:
            new_corr.pyramid = [level[keep] for level in self.pyramid]
        return new_corr

    def __call__(self, coords: torch.Tensor):  # [T, h, w, 2]
        T, ht, wd, _ = coords.shape
        coords = coords.permute(0, 3, 1, 2).contiguous()

        K = (2 * self.radius + 1) ** 2
        out = torch.empty(
            T, self.num_levels * K, ht, wd, device=coords.device, dtype=coords.dtype
        )

        scale = 1
        for i in range(self.num_levels):
            out[:, i * K : (i + 1) * K, :, :] = corr_forward(
                self.pyramid[i], coords / scale, self.radius
            )  # [T, K, h, w]
            scale <<= 1

        return out


class AltCorrBlock:
    def __init__(self, cfg, fmaps):
        self.num_levels = int(cfg.get("altcorr_block", {}).get("num_levels", 4))
        self.radius = int(cfg.get("altcorr_block", {}).get("radius", 3))

        fmaps = fmaps / 4.0
        self.pyramid = []
        for lvl in range(self.num_levels):

            pyramid_level = fmaps.permute(0, 2, 3, 1).contiguous()
            self.pyramid.append(pyramid_level)

            if lvl + 1 < self.num_levels:
                fmaps = F.avg_pool2d(fmaps, kernel_size=2, stride=2)

    def __call__(self, coords: torch.Tensor, ii, jj):
        T, H, W, _ = coords.shape  # [T, H, W, 2]
        coords = coords.permute(0, 3, 1, 2).contiguous()

        K = (2 * self.radius + 1) ** 2
        out = torch.empty(
            T, self.num_levels * K, H, W, device=coords.device, dtype=coords.dtype
        )

        scale = 1
        fmap1 = self.pyramid[0][ii]  # [T, H, W, C]
        for lvl in range(self.num_levels):
            fmap2_lvl = self.pyramid[lvl][jj]  # [T, H//2**i, W//2**i, C]

            coords_lvl = coords / scale
            corr = altcorr_forward(
                fmap1, fmap2_lvl, coords_lvl, self.radius
            )  # [T, K, H, W]
            out[:, lvl * K : (lvl + 1) * K, :, :] = corr
            scale <<= 1

        return out.contiguous()
