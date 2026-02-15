import gsplat
import torch
import torch.nn.functional as F

from geometry import Intrinsics


def _ssim(img1, img2, window_size=11):
    C1 = 0.01**2
    C2 = 0.03**2

    x = img1.permute(0, 3, 1, 2)
    y = img2.permute(0, 3, 1, 2)
    B, Ch = x.shape[:2]

    x = x.reshape(B * Ch, 1, x.shape[2], x.shape[3])
    y = y.reshape(B * Ch, 1, y.shape[2], y.shape[3])

    kernel = torch.ones(1, 1, window_size, window_size, device=x.device) / (
        window_size * window_size
    )

    mu_x = F.conv2d(x, kernel, padding=window_size // 2)
    mu_y = F.conv2d(y, kernel, padding=window_size // 2)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=window_size // 2) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=window_size // 2) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )

    return ssim_map.mean()


class GaussianOptimizer:
    def __init__(self, cfg):
        H_out = int(cfg.get("cam", {}).get("H_out", 480))
        W_out = int(cfg.get("cam", {}).get("W_out", 640))
        fx = float(cfg.get("cam", {}).get("fx", 300.0))
        fy = float(cfg.get("cam", {}).get("fy", 300.0))
        cx = float(cfg.get("cam", {}).get("cx", W_out / 2.0))
        cy = float(cfg.get("cam", {}).get("cy", H_out / 2.0))

        self.H = H_out
        self.W = W_out
        self.K = Intrinsics(fx, fy, cx, cy).as_matrix.unsqueeze(0)

        optim_cfg = cfg.get("splatting", {}).get("optim", {})
        lr_cfg = optim_cfg.get("lr", {})

        self.ssim_weight = float(optim_cfg.get("ssim_weight", 0.2))
        self.opacity_reg = float(optim_cfg.get("opacity_reg", 0.01))
        self.scale_reg = float(optim_cfg.get("scale_reg", 0.02))

        self.lr = {
            "means": float(lr_cfg.get("means", 1e-4)),
            "colors": float(lr_cfg.get("colors", 1e-3)),
            "scales": float(lr_cfg.get("scales", 5e-3)),
            "quats": float(lr_cfg.get("quats", 1e-3)),
            "alphas": float(lr_cfg.get("alphas", 5e-2)),
        }

        self._last_n_gaussians = 0
        self._optimizer = None

    def build_optimizer(self, buffer):
        self._optimizer = torch.optim.AdamW(
            [
                {"params": [buffer.means], "lr": self.lr["means"]},
                {"params": [buffer.colors], "lr": self.lr["colors"]},
                {"params": [buffer.scales], "lr": self.lr["scales"]},
                {"params": [buffer.quats], "lr": self.lr["quats"]},
                {"params": [buffer.alphas], "lr": self.lr["alphas"]},
            ]
        )

    def is_outdated(self, n_gaussians):
        _must_rebuild = self._last_n_gaussians != n_gaussians
        self._last_n_gaussians = n_gaussians
        return _must_rebuild or self._optimizer is None

    def __call__(
        self,
        viewmats,
        gt_colors,
        buffer,
    ):
        colors = buffer.colors.clamp(0, 1)
        scales = buffer.scales.exp()
        opacities = buffer.alphas.sigmoid()
        quats = buffer.quats / buffer.quats.norm(dim=-1, keepdim=True)

        rendered, *_ = gsplat.rasterization(
            means=buffer.means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=self.K.expand(viewmats.shape[0], -1, -1),
            width=self.W,
            height=self.H,
        )

        l1_loss = (rendered - gt_colors).abs().mean()
        ssim_loss = 1.0 - _ssim(rendered, gt_colors)

        loss = (
            (1.0 - self.ssim_weight) * l1_loss
            + self.ssim_weight * ssim_loss
            + self.opacity_reg * opacities.mean()
            + self.scale_reg * scales.mean()
        )

        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        return loss.item()
