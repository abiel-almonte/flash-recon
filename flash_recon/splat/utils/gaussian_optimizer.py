import gsplat
import torch
import torch.nn.functional as F

from geometry import Pose, Intrinsics, quat_to_matrix_cuda

from .gaussian_buffer import GaussianBuffer


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


def _multiview_depth_error(
    poses: Pose,
    depths: torch.Tensor,
    intrinsics: Intrinsics,
    means: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    R = quat_to_matrix_cuda(poses.q)  # [K, 3, 3]
    cam = torch.einsum("kij,nj->kni", R, means) + poses.t[:, None, :]  # [K, N, 3]
    depth_proj = cam[..., 2].clamp(1e-5)  # [K, N] depth value for each key frames

    u = intrinsics.fx * cam[..., 0] / depth_proj + intrinsics.cx  # [K, N]
    v = intrinsics.fy * cam[..., 1] / depth_proj + intrinsics.cy

    valid = (cam[..., 2] > 0) & ((u > 0) & (u < W)) & ((v > 0) & (v < H))

    u = 2 * u / (W - 1) - 1  # normalize grid
    v = 2 * v / (H - 1) - 1
    grid = torch.stack([u, v], dim=-1)  # [K, N, 2]

    depth_map = F.grid_sample(
        depths.unsqueeze(1),  # [K, 1, H, W]
        grid.unsqueeze(2),  # [K, N, 1, 2]
        align_corners=True,
    )  # [K, 1, N, 1]
    depth_map = depth_map.squeeze(1).squeeze(-1)  # [K, N]

    error = (depth_proj - depth_map).abs() / depth_map.clamp(1e-5)
    error = error * valid

    error = error.sum(dim=0)  # [N]
    valid_counts = valid.sum(dim=0).clamp(1)
    return error / valid_counts  # [N]


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
        indices,
        buffer: GaussianBuffer,
    ):
        viewmats = buffer.viewmats[indices]
        gt_colors = buffer.gts[indices]
        gt_depths = buffer.gt_depths[indices]

        colors = buffer.colors.clamp(0, 1)
        scales = buffer.scales.exp()
        opacities = buffer.alphas.sigmoid()
        quats = buffer.quats / buffer.quats.norm(dim=-1, keepdim=True)

        rendered, _, meta = gsplat.rasterization(
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

        if buffer._n_keyframes > 12:  # after slam bootstrapping
            poses = buffer.poses[indices]

            depth_error = _multiview_depth_error(
                poses=poses,
                depths=gt_depths,
                intrinsics=buffer.intrinsics,
                means=buffer.means,
                H=self.H,
                W=self.W,
            )

            depth_loss = (depth_error - 0.05).clamp(0).mean()

        else:
            depth_loss = 0.0

        isotropic_loss = (scales - scales.mean(dim=-1, keepdim=True)).abs().mean()

        loss = (
            (1.0 - self.ssim_weight) * l1_loss
            + self.ssim_weight * ssim_loss
            + 10 * isotropic_loss
            + depth_loss
        )

        loss.backward()

        self._optimizer.step()
        self._optimizer.zero_grad()
        return loss.item(), meta
