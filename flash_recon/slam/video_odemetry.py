import torch

from neural import DroidNet
from geometry import projective_transform, get_meshgrid
from neural import CorrBlock

from .factor_graph import FactorGraph
from .keyframe_buffer import KeyFrameBuffer
from .dsp_optimizer import DSPOptimizer
from .structs import BAType


class VideoOdometry:
    def __init__(self, cfg):
        self._droid = DroidNet(cfg).to(cfg.get("device", "cuda"))
        self._corr = CorrBlock(cfg)
        self._motion_corr = CorrBlock(cfg)
        self._dspo = DSPOptimizer(cfg)

        H_out = int(cfg.get("cam", {}).get("H_out", 480))
        W_out = int(cfg.get("cam", {}).get("W_out", 640))
        down_scale = int(cfg.get("cam", {}).get("down_scale", 8))

        ht = H_out // down_scale
        wd = W_out // down_scale
        y, x = get_meshgrid(ht, wd, "cuda", torch.float)

        self._coords0 = torch.stack([x, y], dim=-1)

    def extract(self, frame):
        fmap = self._droid.apply_fnet(frame)
        net, inp = self._droid.apply_cnet(frame)
        return fmap, net, inp

    def compute_motion(self, prev_fmap, fmap, net, inp):
        corr = self._motion_corr
        corr.pyramid = None
        corr.build_pyramid(prev_fmap[:, [0]].float(), fmap[:, [0]].float())
        feat = corr(self._coords0.unsqueeze(0))
        _, delta, _ = self._droid.apply_update(net, inp, feat)
        return delta.norm(dim=-1).mean().item()

    def __call__(
        self,
        buffer: KeyFrameBuffer,
        graph: FactorGraph,
        steps: int,
        alternate: bool = True,
    ):
        ii, jj = graph.get_edges()

        for step in range(steps):
            # reproject with latest geometry
            poses, disps, intrinsics = buffer.get_geometric_attrs()
            target, _, memory, context = graph.get_flow_attrs()

            coords1, _ = projective_transform(
                poses, disps, intrinsics, ii.reshape(-1), jj.reshape(-1)
            )
            motion = (
                torch.cat([coords1 - graph.coords0, target - coords1], dim=-1)
                .permute(0, 1, 4, 2, 3)
                .clamp(-64.0, 64.0)
            )

            # correlation features
            correlation = graph.corr(coords1)
            memory, delta, weight, damping, upmask = self._droid.apply_update(
                context, memory, correlation, motion, ii, jj
            )
            target = coords1 + delta

            # update factor graph state for next iteration
            unique_ii = torch.unique(ii)
            graph.store_residuals(target, weight, memory, damping, unique_ii)
            buffer.upsample_disps(unique_ii, upmask)

            eta = 0.2 * graph.damping[unique_ii].contiguous() + 1e-7

            # alternating optimization type
            if alternate:
                if step % 2 == 0:
                    ba_type = BAType.POSE_DEPTH
                else:
                    ba_type = BAType.DEPTH_SCALE
            else:
                ba_type = BAType.POSE_DEPTH

            params = {
                "target": target,
                "weight": weight,
                "eta": eta,
                "ii": ii,
                "jj": jj,
                "lm": 1e-5,
                "ep": 1e-2,
                "iters": 2,
            }

            ctx = buffer.create_ba_context(ba_type)
            ctx = self._dspo(ctx, params)
            buffer.apply_ba_result(ctx)
