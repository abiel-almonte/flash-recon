import torch

from optimizer import OptimizationPayload, CallerRole
from geometry import (
    Pose,
    Intrinsics,
	identity_pose,
    pose_inv,
    pose_to_matrix,
)


class KeyFrameBuffer:
	def __init__(self, cfg):
		self.device = cfg.get("device", "cuda")
		self.down_scale = int(cfg.get("cam", {}).get("down_scale", 8))
		H_out = int(cfg.get("cam", {}).get("H_out", 480))
		W_out = int(cfg.get("cam", {}).get("W_out", 640))
		self.ht = H_out // self.down_scale
		self.wd = W_out // self.down_scale
		self.radius = int(cfg.get("tracking", {}).get("frontend", {}).get("radius", 1))
		self.iters = int(cfg.get("optim", {}).get("iters_per_call", 2))
		self.ignore_frames = int(cfg.get("tracking", {}).get("warmup", 1))
		self.capacity = int(cfg.get("tracking", {}).get("buffer", 512))

		# Preallocated state
		self._poses: Pose = identity_pose(self.capacity, device=self.device)
		self._intrinsics: Intrinsics = None
		self._disps = torch.zeros(self.capacity, self.ht, self.wd, device=self.device)
		self._mono_disps = torch.zeros(self.capacity, self.ht, self.wd, device=self.device)
		self._scales = torch.ones(self.capacity, device=self.device)
		self._shifts = torch.zeros(self.capacity, device=self.device)
		self._valid_depth_mask = torch.zeros(self.capacity, self.ht, self.wd, device=self.device, dtype=torch.bool)
		self.needs_update = torch.zeros(self.capacity, device=self.device, dtype=torch.bool)
		self._count = 0

		# Cached graph and derivative data
		self._ii = torch.empty(0, dtype=torch.long, device=self.device)
		self._jj = torch.empty(0, dtype=torch.long, device=self.device)
		self._eta = torch.empty(0, self.ht, self.wd, device=self.device)
		self._cached_edges_T = 0
		self._target = torch.zeros(0, self.ht, self.wd, 2, device=self.device)
		self._weight = torch.ones(0, self.ht, self.wd, 2, device=self.device)

		# Feature attrs
		self.fmaps = torch.zeros(self.capacity, 1, 128, self.ht, self.wd, device=self.device)
		self.nets = torch.zeros(self.capacity, 128, self.ht, self.wd, device=self.device)
		self.inps = torch.zeros(self.capacity, 128, self.ht, self.wd, device=self.device)
		

	def get_scale_shift(self, index):
		return [self._scales[index], self._shifts[index]]
	
	def get_disp(self, index):
		return self._disps[index]
	
	def get_depth(self, index):
		return 1.0 / (self._disps[index].clamp_min(1e-7))
	
	def get_vmask(self, index):
		return self._valid_depth_mask[index]
	
	def get_camera2world(self, index):
		w2c = self._poses[index]
		c2w = pose_to_matrix(pose_inv(w2c))
		return c2w

	def get_flow_attrs(self):
		return self._poses, self._disps, self._intrinsics
	
	def get_feature_attrs(self):
		return self.fmaps, self.nets, self.inps

	def set_intrinsics(self, intrinsics: Intrinsics):
		self._intrinsics = intrinsics

	def confirm_update(self, index_start, index_end):
		self.needs_update[index_start:index_end] = True

	def normalize(self):
		disps = self._disps[:self._count]
		t = self._poses.t[:self._count]

		mean = disps.mean().clamp_min(1e-7)
		disps.div_(mean)
		t.mul_(mean)
		
	def append(
		self,
		pose: Pose,
		disp: torch.Tensor,
		mono_disp: torch.Tensor = None,
		fmap: torch.Tensor = None,
		net: torch.Tensor = None,
		inp: torch.Tensor = None
	)-> None:
		
		idx = self._count
		if idx >= self.capacity:
			raise RuntimeError("KeyFrameBuffer capacity exceeded")
		
		self._poses.t[idx] = pose.t
		self._poses.q[idx] = pose.q
		self._disps[idx] = disp
		self._valid_depth_mask[idx] = disp > 0

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
		self._update_edges_cache()

	def _update_edges_cache(self):
		T = self._count

		if T == self._cached_edges_T:
			return
		
		if T <= 1:
			self._ii = torch.empty(0, dtype=torch.long, device=self.device)
			self._jj = torch.empty(0, dtype=torch.long, device=self.device)
			self._eta = torch.empty(0, self.ht, self.wd, device=self.device)
			self._cached_edges_T = T
			return

		new_ii = []
		new_jj = []
		j = T - 1
		start_i = max(0, j - self.radius)
		for i in range(start_i, j):
			new_ii.append(i)
			new_jj.append(j)

		new_ii = torch.as_tensor(new_ii, device=self.device, dtype=torch.long)
		new_jj = torch.as_tensor(new_jj, device=self.device, dtype=torch.long)
		self._ii = torch.cat([self._ii, new_ii], dim=0) if self._ii.numel() else new_ii
		self._jj = torch.cat([self._jj, new_jj], dim=0) if self._jj.numel() else new_jj

		keyframe_indices = torch.unique(self._ii)
		M = keyframe_indices.numel()
		E = self._ii.numel()

		self._eta = torch.ones((M, self.ht, self.wd), device=self.device) * 0.2
		self._cached_edges_T = T
	
	def update_vmask(self, up = True):
		if up:
			if not self.needs_update.any():
				return
			update_indices, = torch.where(self.needs_update)
		else:
			update_indices = torch.arange(self._count, device=self.device)

		if update_indices.numel() == 0:
			return

		disps = self._disps.index_select(0, update_indices)
		depths = 1.0 / (disps.clamp_min(1e-7))
		...

	def create_dspo_payload(self, role : CallerRole) -> OptimizationPayload:
		assert self._intrinsics is not None, "Intrinsics must be set before creating payload"
		T = self._count
		if T == 0:
			raise RuntimeError("Buffer is empty")

		self._update_edges_cache()

		poses = self._poses[:T]
		disps = self._disps[:T]

		if role == CallerRole.FRONTEND:
			mono_disps = self._mono_disps[:T]
			payload = OptimizationPayload(
				role=role,
				target=self._target,
				weight=self._weight,
				eta=self._eta,
				poses=poses,
				disps=disps,
				intrinsics=self._intrinsics,
				ii=self._ii,
				jj=self._jj,
				mono_disps=mono_disps,
				scales=self._scales[:T],
				shifts=self._shifts[:T],
				valid_depth_mask=self._valid_depth_mask[:T],
				iters=self.iters,
				ignore_frames=self.ignore_frames,
			)
		elif role == CallerRole.BACKEND:
			payload = OptimizationPayload(
				role=role,
				target=self._target,
				weight=self._weight,
				eta=self._eta,
				poses=poses,
				disps=disps,
				intrinsics=self._intrinsics,
				ii=self._ii,
				jj=self._jj,
				iters=self.iters,
				n=T
			)
		elif role == CallerRole.TRAJ_FILLER:
			payload = OptimizationPayload(
				role=role,
				target=self._target,
				weight=self._weight,
				poses=poses,
				disps=disps,
				intrinsics=self._intrinsics,
				ii=self._ii,
				jj=self._jj,
				iters=self.iters
			)
		
		return payload
	
	def update_from_dspo_payload(self, payload: OptimizationPayload):
		"""Update buffer from optimized payload"""
		T = self._count
		self._poses[:T] = payload.poses
		self._disps[:T] = payload.disps
		if payload.scales is not None:
			self._scales[:T] = payload.scales
		if payload.shifts is not None:
			self._shifts[:T] = payload.shifts

	def __len__(self):
		return self._count