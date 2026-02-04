import torch
import torch.nn.functional as F

from .structs import OptimizationPayload, CallerRole
from geometry import (
    Pose,
    Intrinsics,
    identity_pose,
    pose_inv,
    pose_to_matrix,
    depth_filter
)


class KeyFrameBuffer:
    def __init__(self, cfg):
        self.device = cfg.get("device", "cuda")
        self.down_scale = int(cfg.get("cam", {}).get("down_scale", 8))
        H_out = int(cfg.get("cam", {}).get("H_out", 480))
        W_out = int(cfg.get("cam", {}).get("W_out", 640))
        fx = float(cfg.get("cam", {}).get("fx", 300.0))
        fy = float(cfg.get("cam", {}).get("fy", 300.0))
        cx = float(cfg.get("cam", {}).get("cx", W_out / 2.0))
        cy = float(cfg.get("cam", {}).get("cy", H_out / 2.0))
        self.ht = H_out // self.down_scale
        self.wd = W_out // self.down_scale
        self.radius = int(cfg.get("tracking", {}).get("frontend", {}).get("radius", 1))
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
        self._intrinsics: Intrinsics = Intrinsics(fx, fy, cx, cy, device=self.device)
        self._disps = torch.ones(self.capacity, self.ht, self.wd, device=self.device)
        self._disps_up = torch.zeros(self.capacity, H_out, W_out, device=self.device)
        self._mono_disps = torch.zeros(
            self.capacity, self.ht, self.wd, device=self.device
        )
        self._scales = torch.zeros(self.capacity, device=self.device)
        self._shifts = torch.zeros(self.capacity, device=self.device)
        self._valid_depth_mask = torch.zeros(
            self.capacity, H_out, W_out, device=self.device, dtype=torch.bool
        )
        self._valid_depth_mask_small = torch.zeros(
            self.capacity, self.ht, self.wd, device=self.device, dtype=torch.bool
        )
        self.needs_update = torch.zeros(
            self.capacity, device=self.device, dtype=torch.bool
        )
        self._count = 0

        # Feature attrs
        self.fmaps = torch.zeros(
            self.capacity, 1, 128, self.ht, self.wd, device=self.device
        )
        self.nets = torch.zeros(
            self.capacity, 128, self.ht, self.wd, device=self.device
        )
        self.inps = torch.zeros(
            self.capacity, 128, self.ht, self.wd, device=self.device
        )

    def get_scale_shift(self, index):
        return self._scales[index], self._shifts[index]

    def get_disp(self, index):
        return self._disps[index]

    def get_depth(self, index):
        return 1.0 / (self._disps[index].clamp_min(1e-7))

    def get_vmask(self, index):
        return self._valid_depth_mask[index]

    def get_camera2world(self, index):
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
        pose: Pose,
        disp: torch.Tensor,
        mono_disp: torch.Tensor = None,
        fmap: torch.Tensor = None,
        net: torch.Tensor = None,
        inp: torch.Tensor = None,
    ) -> None:

        idx = self._count
        if idx >= self.capacity:
            raise RuntimeError("KeyFrameBuffer capacity exceeded")

        self._poses[idx] = pose
        self._disps[idx] = disp
        self._valid_depth_mask_small[idx] = disp > 0

        if mono_disp is not None:
            self._mono_disps[idx] = mono_disp
        if fmap is not None:
            self.fmaps[idx] = fmap
        if net is not None:
            self.nets[idx] = net
        if inp is not None:
            self.inps[idx] = inp

        self.needs_update[idx] = True
        self._count += 1

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
            disps = self._disps_up
        else:
            disps_to_update = torch.index_select(self._disps, 0, update_indices)
            intrinsics = self._intrinsics
            disps = self._disps

        depths = 1.0 / (disps_to_update.clamp_min(1e-5))
        thresh = self.depth_filter_thresh * depths.flatten(1).mean(dim=-1) # [M]

        count = depth_filter(self._poses, disps, intrinsics, update_indices, thresh)
        depths[count < self.depth_filter_n_views] = torch.nan

        depths_median, _ = depths.flatten(1).nanmedian(dim=-1)
        masks = depths < 3 * depths_median[:, None, None]

        if up:
            self._valid_depth_mask[update_indices] = masks
            self.needs_update[update_indices] = False
        else:
            self._valid_depth_mask_small[update_indices] = masks

    def upsample_disps(self, source_indices, upmask):
        disps = self._disps[source_indices].unsqueeze(-1)
        edges, ht, wd, dim = disps.shape

        disps = disps.permute(0, 3, 1, 2).contiguous()
        mask = upmask.view(edges, 1, 9, 8, 8, ht, wd)
        mask = torch.softmax(mask, dim=2)

        up_disps = F.unfold(disps, kernel_size=(3, 3), padding=(1, 1))
        up_disps = up_disps.view(edges, dim, 9, 1, 1, ht, wd)

        up_disps = torch.sum(mask * up_disps, dim=2, keepdim=False)
        up_disps = up_disps.permute(0, 4, 2, 5, 3, 1).contiguous()
        up_disps = up_disps.reshape(edges, 8 * ht, 8 * wd, dim).squeeze(-1)

        self._disps_up[source_indices] = up_disps
        self.set_needs_update(source_indices)

    def normalize(self):
        disps = self._disps[: self._count]
        t = self._poses.t[: self._count]

        mean = disps.mean().clamp_min(1e-5)
        disps.div_(mean)
        t.mul_(mean)

        self.set_needs_update(slice(0, self._count))

    def create_dspo_payload(self, role: CallerRole) -> OptimizationPayload:
        T = self._count
        if T == 0:
            raise RuntimeError("Buffer is empty")

        poses = self._poses[:T]
        disps = self._disps[:T]

        if role == CallerRole.FRONTEND:
            mono_disps = self._mono_disps[:T]

            self.update_vmask(up=False)
            vmask = self._valid_depth_mask_small[:T]

            self.update_scale_shift(mono_disps, disps, vmask)
            scales = self._scales[:T]
            shifts = self._shifts[:T]

            payload = OptimizationPayload(
                role=role,
                poses=poses,
                disps=disps,
                intrinsics=self._intrinsics,
                mono_disps=mono_disps,
                scales=scales,
                shifts=shifts,
                valid_depth_mask=vmask,
                iters=self.iters,
                ignore_frames=self.ignore_frames,
            )
        elif role == CallerRole.BACKEND:
            payload = OptimizationPayload(
                role=role,
                poses=poses,
                disps=disps,
                intrinsics=self._intrinsics,
                iters=self.iters,
                n=T,
            )
        elif role == CallerRole.TRAJ_FILLER:
            payload = OptimizationPayload(
                role=role,
                poses=poses,
                disps=disps,
                intrinsics=self._intrinsics,
                iters=self.iters,
            )

        return payload

    def update_from_dspo_payload(self, payload: OptimizationPayload):
        T = self._count

        self._poses[:T] = payload.poses
        self._disps[:T] = payload.disps

        if payload.scales is not None:
            self._scales[:T] = payload.scales
        if payload.shifts is not None:
            self._shifts[:T] = payload.shifts

        self.set_needs_update(slice(0, T))

    def __len__(self):
        return self._count
