import torch

from ..slam import SLAMSnapshot, KeyFrame
from .utils import GaussianBuffer, GaussianOptimizer, backproject, deform_gaussians


class Splat:
    def __init__(self, cfg) -> None:
        self.device = cfg.get("device", "cuda")

        splat_cfg = cfg.get("splatting", {})
        self.stride = int(splat_cfg.get("stride", 2))
        self.n_views = int(splat_cfg.get("n_views", 16))
        self.prune_every = int(splat_cfg.get("prune_every", 100))
        self.min_opacity = float(splat_cfg.get("min_opacity", 0.3))
        self.max_scale = float(splat_cfg.get("max_scale", 1.5))

        self.buffer = GaussianBuffer(cfg)
        self.optimizer = GaussianOptimizer(cfg)

        self._step_count = 0
        self._prev_snapshot: SLAMSnapshot | None = None

    @property
    def n_gaussians(self):
        return self.buffer.n_gaussians

    @property
    def snapshot(self):
        snapshot = self.buffer.snapshot
        return snapshot.detach()

    def _deform(self, snapshot: SLAMSnapshot):
        n_kf = min(snapshot.n_keyframes, len(self.buffer))
        valid = self.buffer.kf_ids < n_kf
        if not valid.any():
            return

        new_means, new_scales, new_quats = deform_gaussians(
            means=self.buffer.means[valid],
            scales=self.buffer.scales[valid],
            quats=self.buffer.quats[valid],
            pixel_uv=self.buffer.pixel_uv[valid],
            kf_ids=self.buffer.kf_ids[valid],
            old_poses=self._prev_snapshot.poses[:n_kf],
            new_poses=snapshot.poses[:n_kf],
            old_disps=self._prev_snapshot.disps[:n_kf],
            new_disps=snapshot.disps[:n_kf],
        )

        with torch.no_grad():
            self.buffer.means[valid] = new_means
            self.buffer.scales[valid] = new_scales
            self.buffer.quats[valid] = new_quats

    def _should_prune(self):
        self._step_count += 1
        return self._step_count % self.prune_every == 0

    def _step(self):
        if self.buffer.n_gaussians == 0:
            return 0.0

        if self.optimizer.is_outdated(self.buffer.n_gaussians):
            self.buffer.reset_grad()
            self.optimizer.build_optimizer(self.buffer)

        indices = torch.randperm(len(self.buffer))[: self.n_views]

        viewmats = self.buffer.viewmats[indices]
        gt_colors = self.buffer.gts[indices]

        loss = self.optimizer(
            viewmats=viewmats,
            gt_colors=gt_colors,
            buffer=self.buffer,
        )

        self.buffer.clip()

        if self._should_prune():
            self.buffer.prune(self.min_opacity, self.max_scale)

        return loss

    def add(self, keyframe: KeyFrame):
        result = backproject(
            frame=keyframe.frame,
            pose=keyframe.poses,
            disps=keyframe.disps,
            vmask=keyframe.vmask,
            intrinsics=self.buffer.intrinsics,
            stride=self.stride,
        )

        if result is None:
            return False

        world, colors, scales, viewmat, pixel_uv = result
        self.buffer.append(keyframe.frame, viewmat, world, colors, scales, pixel_uv)

        return True

    def __call__(self, snapshot: SLAMSnapshot):
        if (
            self._prev_snapshot is not None
            and snapshot.version != self._prev_snapshot.version
        ):  # if keyframe buffer state changed
            self._deform(snapshot)

        self._prev_snapshot = snapshot.copy()

        return self._step()
