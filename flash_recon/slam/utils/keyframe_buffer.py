from typing import Optional

import torch
import torch.nn.functional as F

from geometry import (
    Pose,
    Intrinsics,
    identity_pose,
    pose_inv,
    pose_to_matrix,
    depth_filter,
)

from .contexts import BAContext, BufferSnapshot, KeyFrame
from .enums import BAType


class KeyFrameBuffer:
    def __init__(self, cfg):
        self.device = cfg.get("device", "cuda")
        down = int(cfg.get("cam", {}).get("down_scale", 8))
        self.down_scale = down

        H_out = int(cfg.get("cam", {}).get("H_out", 480))
        W_out = int(cfg.get("cam", {}).get("W_out", 640))
        fx = float(cfg.get("cam", {}).get("fx", 300.0))
        fy = float(cfg.get("cam", {}).get("fy", 300.0))
        cx = float(cfg.get("cam", {}).get("cx", W_out / 2.0))
        cy = float(cfg.get("cam", {}).get("cy", H_out / 2.0))
        ht = H_out // down
        wd = W_out // down

        self.iters = int(cfg.get("optim", {}).get("iters_per_call", 2))
        self.ignore_frames = int(cfg.get("tracking", {}).get("warmup", 1))
        self.capacity = int(cfg.get("tracking", {}).get("buffer", 512))
        self.depth_filter_thresh = (
            cfg.get("tracking", {}).get("depth_filter", {}).get("thresh", 0.01)
        )
        self.depth_filter_n_views = (
            cfg.get("tracking", {}).get("depth_filter", {}).get("n_views", 2)
        )

        # Preallocated state
        self._poses: Pose = identity_pose(self.capacity, device=self.device)
        self._intrinsics: Intrinsics = Intrinsics(
            fx, fy, cx, cy, device=self.device
        ).downsample(down)
        self._disps = torch.ones(self.capacity, ht, wd, device=self.device)
        self._disps_up = torch.zeros(self.capacity, H_out, W_out, device=self.device)
        self._mono_depths = torch.zeros(self.capacity, ht, wd, device=self.device)
        self._scales = torch.zeros(self.capacity, device=self.device)
        self._shifts = torch.zeros(self.capacity, device=self.device)
        self._valid_depth_mask = torch.zeros(
            self.capacity, H_out, W_out, device=self.device, dtype=torch.bool
        )
        self._valid_depth_mask_small = torch.zeros(
            self.capacity, ht, wd, device=self.device, dtype=torch.bool
        )
        self.needs_update = torch.zeros(
            self.capacity, device=self.device, dtype=torch.bool
        )
        self._count = 0
        self._version = 0

        # Feature attrs
        self.fmaps = torch.zeros(self.capacity, 1, 128, ht, wd, device=self.device)
        self.nets = torch.zeros(self.capacity, 128, ht, wd, device=self.device)
        self.inps = torch.zeros(self.capacity, 128, ht, wd, device=self.device)

    @property
    def version(self):
        return self._version

    def get_keyframe(self, idx: int) -> KeyFrame:
        if not (0 <= idx < self._count):
            raise IndexError(
                f"Keyframe index {idx} out of range (n_keyframes={self._count})"
            )

        return KeyFrame(
            poses=self._poses[idx],
            disps=self._disps_up[idx],
            vmask=self._valid_depth_mask[idx],
        )

    def get_scale_shift(self, index):
        return self._scales[index], self._shifts[index]

    def get_disp(self, index):
        return self._disps[index]

    def get_depth(self, index):
        return 1.0 / (self._disps[index].clamp_min(1e-7))

    def get_vmask(self, index: Optional[int] = None):
        if index is not None:
            return self._valid_depth_mask[index]
        else:
            return self._valid_depth_mask

    def get_cam2world(self, index):
        w2c = self._poses[index]
        c2w = pose_to_matrix(pose_inv(w2c))
        return c2w.squeeze(0)

    def get_geometric_attrs(self):
        return self._poses, self._disps, self._intrinsics

    def get_neural_attrs(self):
        return self.fmaps, self.nets, self.inps

    def set_intrinsics(self, intrinsics: Intrinsics):
        self._intrinsics = intrinsics

    def set_needs_update(self, indices: torch.Tensor):
        self.needs_update[indices] = True

    def append(
        self,
        pose: Pose = None,
        disp: torch.Tensor = None,
        mono_depth: torch.Tensor = None,
        fmap: torch.Tensor = None,
        net: torch.Tensor = None,
        inp: torch.Tensor = None,
    ) -> None:

        idx = self._count
        if idx >= self.capacity:
            raise RuntimeError("KeyFrameBuffer capacity exceeded")

        if pose is not None:
            self._poses[idx] = pose
        if disp is not None:
            self._disps[idx] = disp
        self._valid_depth_mask_small[idx] = self._disps[idx] > 0

        if mono_depth is not None:
            mono_disp = torch.where(
                mono_depth > 0, 1.0 / mono_depth, torch.zeros_like(mono_depth)
            )
            self._mono_depths[idx] = mono_disp
        if fmap is not None:
            self.fmaps[idx] = fmap
        if net is not None:
            self.nets[idx] = net
        if inp is not None:
            self.inps[idx] = inp

        self.needs_update[idx] = True
        self._count += 1

    def propagate(self, use_init_mean=False):
        idx = self._count
        if idx > 0 and idx < self.capacity:
            self._poses[idx] = self._poses[idx - 1]
            if use_init_mean:
                self._disps[idx] = self._disps[max(0, idx - 4) : idx].mean()
            else:
                self._disps[idx] = self._disps[idx - 1].mean()

    def remove(self, idx: int) -> None:
        if idx < 0 or idx >= self._count:
            raise IndexError(f"Index {idx} out of range [0, {self._count})")

        if idx < self._count - 1:
            src = slice(idx + 1, self._count)
            dst = slice(idx, self._count - 1)

            self._poses[dst] = self._poses[src]
            self._disps[dst] = self._disps[src]
            self._disps_up[dst] = self._disps_up[src]
            self._mono_depths[dst] = self._mono_depths[src]
            self._scales[dst] = self._scales[src]
            self._shifts[dst] = self._shifts[src]
            self._valid_depth_mask[dst] = self._valid_depth_mask[src]
            self._valid_depth_mask_small[dst] = self._valid_depth_mask_small[src]
            self.needs_update[dst] = self.needs_update[src]
            self.fmaps[dst] = self.fmaps[src]
            self.nets[dst] = self.nets[src]
            self.inps[dst] = self.inps[src]

        self._count -= 1

    def update_scale_shift(
        self,
        mono: torch.Tensor,
        disps: torch.Tensor,
        weights: torch.Tensor,
        eps: float = 1e-7,
    ):
        T = mono.size(0)
        x = mono.view(T, -1)
        y = disps.view(T, -1)
        w = weights.view(T, -1)

        wx = w * x
        wy = w * y

        a00 = torch.sum(wx * x, dim=1)
        a01 = torch.sum(wx, dim=1)
        a11 = torch.sum(w, dim=1)
        b0 = torch.sum(wx * y, dim=1)
        b1 = torch.sum(wy, dim=1)

        det = a00 * a11 - a01 * a01 + eps
        scale = (a11 * b0 - a01 * b1) / det
        shift = (a00 * b1 - a01 * b0) / det

        self._scales[:T] = scale
        self._shifts[:T] = shift

    def update_vmask(self, up=True):
        if up:
            if not self.needs_update.any():
                return
            (update_indices,) = torch.where(self.needs_update)
        else:
            update_indices = torch.arange(self._count, device=self.device)

        if update_indices.numel() == 0:
            return

        if up:
            disps_to_update = torch.index_select(self._disps_up, 0, update_indices)
            intrinsics = self._intrinsics.scale_resolution(self.down_scale)
            disps = self._disps_up[: self._count]
        else:
            disps_to_update = torch.index_select(self._disps, 0, update_indices)
            intrinsics = self._intrinsics
            disps = self._disps[: self._count]

        depths = 1.0 / (disps_to_update.clamp_min(1e-5))
        thresh = self.depth_filter_thresh * depths.flatten(1).mean(dim=-1)  # [M]

        poses = self._poses[: self._count]
        count = depth_filter(poses, disps, intrinsics, update_indices, thresh)
        depths[count < self.depth_filter_n_views] = torch.nan

        depths_median, _ = depths.flatten(1).nanmedian(dim=-1)
        masks = depths < 3 * depths_median[:, None, None]

        if up:
            self._valid_depth_mask[update_indices] = masks
            self.needs_update[update_indices] = False
        else:
            self._valid_depth_mask_small[update_indices] = masks

    def upsample_disps(self, source_indices, upmask):
        disps = self._disps[source_indices]

        self._disps_up[source_indices] = F.interpolate(
            disps.unsqueeze(1),
            scale_factor=self.down_scale,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        self.set_needs_update(source_indices)

    def normalize(self):
        disps = self._disps[: self._count]
        t = self._poses.t[: self._count]

        mean = disps.mean().clamp_min(1e-5)

        self._disps[: self._count] = disps.div(mean)
        self._poses.t[: self._count] = t.mul(mean)

        self.set_needs_update(slice(0, self._count))

    def create_ba_context(self, ba_type: BAType) -> BAContext:
        T = self._count
        if T == 0:
            raise RuntimeError("Buffer is empty")

        poses = self._poses[:T]
        disps = self._disps[:T]

        if ba_type == BAType.POSE_DEPTH:
            return BAContext(
                type=ba_type,
                poses=poses,
                disps=disps,
                intrinsics=self._intrinsics,
                iters=self.iters,
                n=T,
            )

        elif ba_type == BAType.DEPTH_SCALE:
            mono_depths = self._mono_depths[:T]

            self.update_vmask(up=False)
            vmask = self._valid_depth_mask_small[:T]

            self.update_scale_shift(mono_depths, disps, vmask)
            scales = self._scales[:T]
            shifts = self._shifts[:T]

            # mono filtering
            fitted = scales[:, None, None] * mono_depths + shifts[:, None, None]
            error = ((fitted - disps).abs() * vmask).sum(dim=[1, 2]) / vmask.sum(
                dim=[1, 2]
            ).clamp_min(1)
            avg_disps = disps.mean(dim=[1, 2])
            invalid_mono = (
                (error / avg_disps.clamp_min(1e-7) > 0.1)
                | error.isnan()
                | (scales < 0)
                | (vmask.sum(dim=[1, 2]) < vmask.shape[1] * vmask.shape[2] * 0.5)
            )

            return BAContext(
                type=ba_type,
                poses=poses,
                disps=disps,
                intrinsics=self._intrinsics,
                mono_depths=mono_depths,
                scales=scales,
                shifts=shifts,
                valid_depth_mask=vmask,
                invalid_mono_frames=invalid_mono,
                iters=self.iters,
                ignore_frames=0,
            )

        elif ba_type == BAType.MOTION_ONLY:
            return BAContext(
                type=ba_type,
                poses=poses,
                disps=disps,
                intrinsics=self._intrinsics,
                iters=self.iters,
            )

        return None

    def apply_ba_result(self, ctx: BAContext):
        T = self._count

        self._poses[:T] = ctx.poses
        self._disps[:T] = ctx.disps
        self._disps[:T].clamp_(min=1e-5)

        if ctx.scales is not None:
            self._scales[:T] = ctx.scales
        if ctx.shifts is not None:
            self._shifts[:T] = ctx.shifts

        self.set_needs_update(slice(0, T))
        self._version += 1

    @property
    def snapshot(self):
        return BufferSnapshot(
            count=self._count,
            poses=self._poses,
            disps=self._disps,
            intrinsics=self._intrinsics,
            fmaps=self.fmaps,
            nets=self.nets,
            inps=self.inps,
        )

    def __len__(self):
        return self._count
