import torch

from geometry import Pose, Intrinsics

from .structs import SplatSnapshot


class GaussianBuffer:
    def __init__(self, cfg):
        device = cfg.get("device", "cuda")

        H_out = int(cfg.get("cam", {}).get("H_out", 480))
        W_out = int(cfg.get("cam", {}).get("W_out", 640))
        fx = float(cfg.get("cam", {}).get("fx", 300.0))
        fy = float(cfg.get("cam", {}).get("fy", 300.0))
        cx = float(cfg.get("cam", {}).get("cx", W_out / 2.0))
        cy = float(cfg.get("cam", {}).get("cy", H_out / 2.0))

        self.device = device
        self.H = H_out
        self.W = W_out
        self.intrinsics = Intrinsics(fx, fy, cx, cy)

        self.means = torch.empty(0, 3, device=device)
        self.colors = torch.empty(0, 3, device=device)
        self.scales = torch.empty(0, 3, device=device)
        self.quats = torch.empty(0, 4, device=device)
        self.alphas = torch.empty(0, device=device)
        self.pixel_uv = torch.empty(0, 2, device=device, dtype=torch.long)
        self.kf_ids = torch.empty(0, device=device, dtype=torch.long)

        self.poses: Pose | None = None
        self.gts = torch.empty(0, H_out, W_out, 3, device=device)
        self.gt_depths = torch.empty(0, H_out, W_out, device=device)
        self.viewmats = torch.empty(0, 4, 4, device=device)
        self.n_touched = torch.zeros(0, device=device)
        self._n_keyframes = 0

    @property
    def n_gaussians(self):
        return self.means.shape[0]

    @property
    def snapshot(self):
        return SplatSnapshot(
            n=self.n_gaussians,
            means=self.means,
            colors=self.colors,
            scales=self.scales,
            quats=self.quats,
            alphas=self.alphas,
        )

    def accumulate_touched(self, n_touched):
        self.n_touched = self.n_touched + n_touched

    def reset_grad(self):
        self.means = self.means.detach().requires_grad_(True)
        self.colors = self.colors.detach().requires_grad_(True)
        self.scales = self.scales.detach().requires_grad_(True)
        self.quats = self.quats.detach().requires_grad_(True)
        self.alphas = self.alphas.detach().requires_grad_(True)

    def append(self, gt, gt_depth, poses, viewmat, means, colors, scales, pixel_uv):
        kf_idx = self._n_keyframes
        n = means.shape[0]

        if self.poses is None:
            self.poses = poses
        else:
            self.poses = self.poses.concatenate(poses)

        self.means = torch.cat([self.means, means], dim=0)
        self.colors = torch.cat([self.colors, colors], dim=0)
        self.scales = torch.cat([self.scales, scales], dim=0)
        self.pixel_uv = torch.cat([self.pixel_uv, pixel_uv], dim=0)
        self.kf_ids = torch.cat(
            [
                self.kf_ids,
                torch.full((n,), kf_idx, device=self.device, dtype=torch.long),
            ],
            dim=0,
        )
        self.quats = torch.cat(
            [
                self.quats,
                torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float)
                .unsqueeze(0)
                .expand(n, -1),
            ],
            dim=0,
        )
        self.alphas = torch.cat(
            [self.alphas, torch.full((n,), 2.0, device=self.device)], dim=0
        )

        self.gts = torch.cat([self.gts, gt.unsqueeze(0)], dim=0)
        self.gt_depths = torch.cat([self.gt_depths, gt_depth.unsqueeze(0)], dim=0)
        self.viewmats = torch.cat(
            [self.viewmats, viewmat.squeeze(0).unsqueeze(0)], dim=0
        )
        self.n_touched = torch.cat(
            [self.n_touched, torch.zeros((n,), device=self.device)], dim=0
        )

        self._n_keyframes += 1

    def prune(self, min_opacity=0.01, max_scale=None):
        if self.poses is None or self._n_keyframes < 2:
            return

        with torch.no_grad():
            mask = self.alphas.sigmoid() > min_opacity

            if max_scale is not None:
                mask = mask & (self.scales.exp().max(dim=-1).values < max_scale)

            old_enough = self.kf_ids < (self._n_keyframes - 10)
            mask = mask & (~old_enough | (self.n_touched >= 60))

            n_before = self.means.shape[0]
            if mask.sum() < n_before:
                self.means = self.means[mask]
                self.colors = self.colors[mask]
                self.scales = self.scales[mask]
                self.quats = self.quats[mask]
                self.alphas = self.alphas[mask]
                self.pixel_uv = self.pixel_uv[mask]
                self.kf_ids = self.kf_ids[mask]
                self.n_touched = self.n_touched[mask]

            self.n_touched.zero_()

    def clip(self):
        with torch.no_grad():
            self.scales.clamp_(-7.0, -2.0)
            self.alphas.clamp_(-5.0, 5.0)

    def __len__(self):
        return self._n_keyframes
