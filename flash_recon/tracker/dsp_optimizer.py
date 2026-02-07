from dataclasses import replace

from geometry import full_ba, ba_scale_shift, motion_only_ba
from .structs import BAContext, BAType


class DSPOptimizer:
    def __init__(self, cfg):
        optim_cfg = cfg.get("optim", {}) if isinstance(cfg, dict) else {}
        self.num_fixed_poses = optim_cfg.get("num_fixed_poses", 1)
        self.rig_size = optim_cfg.get("rig_size", 1)
        self.lm = optim_cfg.get("lm", 1e-4)
        self.ep = optim_cfg.get("ep", 0.1)
        self.alpha = optim_cfg.get("alpha", 0.05)

    def _dispatcher(self, ctx: BAContext) -> BAContext:
        """Dispatch optimization schedule based on the ba type.

        ba type: {"pose_depth", "depth_scale", "motion_only"}
        """
        iters = max(1, ctx.iters)
        type = ctx.type if ctx.type else BAType.POSE_DEPTH

        if type is BAType.POSE_DEPTH:
            for _ in range(iters):
                self._step_pose_depth(ctx)
        
        elif type is BAType.DEPTH_SCALE:
            for _ in range(iters):
                self._step_depth_scale(ctx)

        elif type is BAType.MOTION_ONLY:
            for _ in range(iters):
                self._step_motion_only(ctx)

        return ctx

    def __call__(
        self, ctx: BAContext, params: dict = None
    ) -> BAContext:
        if params:
            ctx = replace(ctx, **params)

        if ctx.lm is None:
            ctx = replace(ctx, lm=self.lm)
        if ctx.ep is None:
            ctx = replace(ctx, ep=self.ep)
        if ctx.alpha is None:
            ctx = replace(ctx, alpha=self.alpha)
        if ctx.num_fixed_poses is None:
            ctx = replace(ctx, num_fixed_poses=self.num_fixed_poses)
        if ctx.rig_size is None:
            ctx = replace(ctx, rig_size=self.rig_size)

        return self._dispatcher(ctx)

    def _step_depth_scale(self, ctx: BAContext) -> BAContext:
        """Perform depth-scale optimization (full BA w/ scale shift)  step."""
        disps_out, scale_shift = ba_scale_shift(
            ctx.target,
            ctx.weight,
            ctx.eta,
            ctx.poses,
            ctx.disps,
            ctx.intrinsics,
            ctx.ii,
            ctx.jj,
            ctx.mono_depths,
            ctx.scales,
            ctx.shifts,
            ctx.valid_depth_mask,
            ctx.ignore_frames,
            ctx.lm,
            ctx.ep,
            ctx.alpha,
        )
        ctx.disps = disps_out
        ctx.scales = scale_shift[:, 0]
        ctx.shifts = scale_shift[:, 1]
        return ctx

    def _step_pose_depth(self, ctx: BAContext) -> BAContext:
        """Perform pose-depth (full BA) optimization step."""
        poses_out, disps_out = full_ba(
            ctx.target,
            ctx.weight,
            ctx.eta,
            ctx.poses,
            ctx.disps,
            ctx.intrinsics,
            ctx.ii,
            ctx.jj,
            ctx.n,
            ctx.lm,
            ctx.ep,
            ctx.alpha,
            ctx.num_fixed_poses,
            ctx.rig_size,
        )
        ctx.poses = poses_out
        ctx.disps = disps_out
        return ctx

    def _step_motion_only(self, ctx: BAContext) -> BAContext:
        """Perform motion-only BA step (poses only)."""
        poses_out = motion_only_ba(
            ctx.target,
            ctx.weight,
            ctx.poses,
            ctx.disps,
            ctx.intrinsics,
            ctx.ii,
            ctx.jj,
            ctx.num_fixed_poses,
            ctx.rig_size,
            ctx.lm,
            ctx.ep,
        )
        ctx.poses = poses_out
        return ctx
