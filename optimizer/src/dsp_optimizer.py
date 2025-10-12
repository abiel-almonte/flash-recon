from structs import OptimizationPayload, CallerRole
from geometry import full_ba, motion_only_ba, ba_scale_shift

class DSPOptimizer:
	def __init__(self, cfg):
		optim_cfg = cfg.get("optim", {}) if isinstance(cfg, dict) else {}
		self.ds_iters = optim_cfg.get("depth_scales_iters", 1)
		self.pd_iters = optim_cfg.get("poses_depth_iters", 1)
		self.num_fixed_poses = optim_cfg.get("num_fixed_poses", 1)
		self.rig_size = optim_cfg.get("rig_size", 1)
		self.lm = optim_cfg.get("lm", 1e-4)
		self.ep = optim_cfg.get("ep", 0.1)
		self.alpha = optim_cfg.get("alpha", 0.05)

	def _dispatcher(self, payload: OptimizationPayload, caller: str) -> OptimizationPayload:
		"""Dispatch optimization schedule based on the caller role.

		caller: one of {"frontend", "backend", "traj_filler"}
		"""
		iters = max(1, payload.iters)
		role = CallerRole(caller)

		if role is CallerRole.FRONTEND:
			for itr in range(iters):

				if itr % 2 == 0:
					for _ in range(self.pd_iters):
						self._step_pose_depth(payload)
				else:
					for _ in range(self.ds_iters):
						self._step_depth_scale(payload)

		elif role is CallerRole.BACKEND:
			for _ in range(iters):

				for _ in range(self.pd_iters):
					self._step_pose_depth(payload)

				for _ in range(self.ds_iters):
					self._step_depth_scale(payload)

		elif role is CallerRole.TRAJ_FILLER:
			for _ in range(iters):
				self._step_motion_only(payload)

		else:
			for _ in range(iters):
				for _ in range(self.pd_iters):
					self._step_pose_depth(payload)
					
				for _ in range(self.ds_iters):
					self._step_depth_scale(payload)

		return payload

	def __call__(self, buffer, caller: str = "frontend") -> OptimizationPayload:
		payload = buffer.create_dspo_payload(caller)

		if payload.lm is None:
			payload.lm = self.lm
		if payload.ep is None:
			payload.ep = self.ep
		if payload.alpha is None:
			payload.alpha = self.alpha
		if payload.num_fixed_poses is None:
			payload.num_fixed_poses = self.num_fixed_poses
		if payload.rig_size is None:
			payload.rig_size = self.rig_size

		return self._dispatcher(payload, caller)

	def _step_depth_scale(self, payload: OptimizationPayload) -> OptimizationPayload:
		"""Perform depth-scale optimization (full BA w/ scale shift)  step."""
		disps_out, scale_shift = ba_scale_shift(
			payload.target,
			payload.weight,
			payload.eta,
			payload.poses,
			payload.disps,
			payload.intrinsics,
			payload.ii,
			payload.jj,
			payload.mono_disps,
			payload.scales,
			payload.shifts,
			payload.valid_depth_mask,
			payload.ignore_frames,
			payload.lm,
			payload.ep,
			payload.alpha,
		)
		payload.disps = disps_out
		payload.scales = scale_shift[:, 0]
		payload.shifts = scale_shift[:, 1]
		return payload

	def _step_pose_depth(self, payload: OptimizationPayload) -> OptimizationPayload:
		"""Perform pose-depth (full BA) optimization step."""
		poses_out, disps_out = full_ba(
			payload.target,
			payload.weight,
			payload.eta,
			payload.poses,
			payload.disps,
			payload.intrinsics,
			payload.ii,
			payload.jj,
			payload.n,
			payload.lm,
			payload.ep,
			payload.alpha,
			payload.num_fixed_poses,
			payload.rig_size,
		)
		payload.poses = poses_out
		payload.disps = disps_out
		return payload

	def _step_motion_only(self, payload: OptimizationPayload) -> OptimizationPayload:
		"""Perform motion-only BA step (poses only)."""
		poses_out = motion_only_ba(
			payload.target,
			payload.weight,
			payload.poses,
			payload.disps,
			payload.intrinsics,
			payload.ii,
			payload.jj,
			payload.num_fixed_poses,
			payload.rig_size,
			payload.lm,
			payload.ep,
		)
		payload.poses = poses_out
		return payload


