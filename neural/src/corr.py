import torch
import torch.nn.functional as F

from neural_cuda.corr import corr_forward


class CorrBlock:

    def __init__(self, num_levels: int = 4, radius: int = 3) -> None:
        super().__init__()

        self.num_levels = num_levels
        self.radius = radius
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

        self.pyramid = []
        for i in range(self.num_levels):
            Hi = corr.shape[-2]
            Wi = corr.shape[-1]

            self.pyramid.append(corr.view(T, ht, wd, Hi, Wi))

            if i + 1 < self.num_levels:
                corr = F.avg_pool2d(corr, kernel_size=2, stride=2)

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
