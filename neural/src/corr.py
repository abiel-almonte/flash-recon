import torch
import torch.nn.functional as F

from neural_cuda.corr import corr_forward, altcorr_forward


class Corr:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.num_levels = int(cfg.get("corr_block", {}).get("num_levels", 4))
        self.radius = int(cfg.get("corr_block", {}).get("radius", 3))
        self.max_factors = int(
            cfg.get("corrblock", {}).get(
                "max_slots", int(cfg.get("tracking", {}).get("max_factors", 512))
            )
        )
        self.pyramid = None  # Pre-allocated: [max_factors, ht, wd, Hi, Wi] per level
        self.num_edges = 0

    def _preallocate_pyramid(self, ht, wd, device):
        self.pyramid = []
        Hi, Wi = ht, wd
        for _ in range(self.num_levels):
            self.pyramid.append(
                torch.zeros(
                    self.max_factors, ht, wd, Hi, Wi, dtype=torch.half, device=device
                )
            )
            Hi = (Hi + 1) // 2
            Wi = (Wi + 1) // 2
        self.num_edges = 0

    @torch.autocast("cuda", enabled=True)
    @torch.no_grad()
    def build_pyramid(
        self,
        feature1: torch.Tensor,
        feature2: torch.Tensor,
    ):  # features will be [T, c, h, w] each
        T, c, ht, wd = feature1.shape
        p = ht * wd

        if self.pyramid is None:
            self._preallocate_pyramid(ht, wd, feature1.device)

        f1 = feature1.reshape(T, c, p) / 4.0
        f2 = feature2.reshape(T, c, p) / 4.0

        corr = torch.bmm(f1.transpose(-2, -1), f2)  # [T, p, p]
        corr = corr.half()
        corr = corr.view(T * p, 1, ht, wd)

        start = self.num_edges
        end = start + T

        for i in range(self.num_levels):
            Hi = corr.shape[-2]
            Wi = corr.shape[-1]

            self.pyramid[i][start:end] = corr.view(T, ht, wd, Hi, Wi)

            if i + 1 < self.num_levels:
                corr = F.avg_pool2d(corr, kernel_size=2, stride=2)

        self.num_edges = end

    def clear_pyramid(self):
        self.num_edges = 0

    def filter_pyramid(self, keep: torch.Tensor):
        if self.pyramid is None or self.num_edges == 0:
            return
        n_keep = keep.sum().item()
        for i in range(self.num_levels):
            self.pyramid[i][:n_keep] = self.pyramid[i][: self.num_edges][keep]
        self.num_edges = n_keep

    def __call__(self, coords: torch.Tensor):  # [T, h, w, 2]
        T, H, W, _ = coords.shape
        coords = coords.permute(0, 3, 1, 2).contiguous()

        K = (2 * self.radius + 1) ** 2
        out = torch.empty(
            T, self.num_levels * K, H, W, device=coords.device, dtype=coords.dtype
        )

        scale = 1
        for lvl in range(self.num_levels):
            coords_lvl = coords / scale
            corr = corr_forward(
                self.pyramid[lvl][: self.num_edges], coords_lvl, self.radius
            )  # [T, K, H, W]
            out[:, lvl * K : (lvl + 1) * K, :, :] = corr
            scale <<= 1

        return out


class AltCorr:
    def __init__(self, cfg):
        self.num_levels = int(cfg.get("altcorr_block", {}).get("num_levels", 4))
        self.radius = int(cfg.get("altcorr_block", {}).get("radius", 3))
        self.pyramid = None

    def build_pyramid(self, fmaps):
        fmaps = fmaps / 4.0

        self.pyramid = []
        for lvl in range(self.num_levels):

            pyramid_level = fmaps.permute(0, 2, 3, 1).contiguous()
            self.pyramid.append(pyramid_level)

            if lvl + 1 < self.num_levels:
                fmaps = F.avg_pool2d(fmaps, kernel_size=2, stride=2)

    def clear(self):
        self.pyramid = None

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
