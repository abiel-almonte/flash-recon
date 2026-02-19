from typing import Tuple, Optional, Union

import torch

from neural import DroidNet, AltCorr
from geometry import projective_transform, get_meshgrid

from .utils import (
    FactorGraph,
    LocalGraph,
    GlobalGraph,
    KeyFrameBuffer,
    DSPOptimizer,
    BAContext,
    BAType,
)


class VideoOdometry:
    def __init__(self, cfg):
        device = cfg.get("device", "cuda")
        self._droid = DroidNet(cfg).to(device)
        self._altcorr = AltCorr(cfg)
        self._dspo = DSPOptimizer(cfg)

        down = int(cfg["cam"].get("down_scale", 8))
        ht = int(cfg["cam"]["H_out"]) // down
        wd = int(cfg["cam"]["W_out"]) // down

        y, x = get_meshgrid(ht, wd, device, torch.float)
        self._coords0 = torch.stack([x, y], dim=-1)
        self.ii0 = torch.tensor([0], device=device, dtype=torch.long)
        self.jj0 = torch.tensor([1], device=device, dtype=torch.long)

    @torch.inference_mode()
    def extract(self, frame: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        fmap = self._droid.apply_fnet(frame)
        net, inp = self._droid.apply_cnet(frame)
        return fmap, net.tanh(), inp.relu()

    @torch.inference_mode()
    def compute_motion(self, prev_fmap, fmap, net, inp) -> torch.Tensor:
        fmaps = torch.cat([prev_fmap, fmap], dim=0)
        self._altcorr.build_pyramid(fmaps)

        corr = self._altcorr(self._coords0.unsqueeze(0), self.ii0, self.jj0)
        _, delta, _ = self._droid.apply_update(net, inp, corr)

        self._altcorr.clear()  # gc pyramid
        return float(delta.norm(dim=-1).mean().item())

    @torch.autocast("cuda", enabled=True)
    @torch.no_grad()
    def _local_update(
        self,
        buffer: KeyFrameBuffer,
        graph: LocalGraph,
        steps: int = 1,
        alternate: bool = True,
        t0: Optional[int] = None,
    ) -> None:
        for step in range(steps):
            ii, jj = graph.get_edges()

            # reproject + motion
            with torch.autocast("cuda", enabled=False):
                poses, disps, intrinsics = buffer.get_geometric_attrs()
                coords1, _ = projective_transform(poses, disps, intrinsics, ii, jj)

                target, _ = graph.get_flow_attrs()
                motn = (
                    torch.cat([coords1 - self._coords0, target - coords1], dim=-1)
                    .permute(0, 3, 1, 2)
                    .clamp(-64.0, 64.0)
                )

            # correlate + network update
            corr_feats = graph.apply_corr(coords1)
            net_hidden, inp = graph.get_neural_attrs()

            net, delta, weight, damping, upmask = self._droid.apply_update(
                net_hidden, inp, corr_feats, motn, ii, jj
            )

            _t0 = t0 if t0 is not None else max(1, ii.min().item() + 1)

            # BA + upsample
            target = coords1 + delta.float()
            weight = weight.float()
            unique_ii = torch.unique(ii)

            graph.update_damping(unique_ii, damping)

            ii_ba, jj_ba, target_ba, weight_ba = graph.merge_inactive(
                ii, jj, target, weight, _t0
            )

            damping = graph.get_damping(torch.unique(ii_ba))
            eta = 0.2 * damping + 1e-7

            ctx, params = self._get_ba_params(
                buffer=buffer,
                step=step,
                alternate=alternate,
                ii=ii_ba,
                jj=jj_ba,
                target=target_ba,
                weight=weight_ba,
                eta=eta,
                t0=_t0,
                lm=1e-4,
                ep=0.1,
            )

            if params is not None:
                result = self._dspo(ctx, params)
                buffer.apply_ba_result(result)

            buffer.upsample_disps(unique_ii, upmask)
            graph.update_residuals(target, weight, net)
            graph.age_edges()

    @torch.autocast("cuda", enabled=False)
    @torch.no_grad()
    def _global_update(
        self,
        buffer: KeyFrameBuffer,
        graph: GlobalGraph,
        t0: int,
        steps: int = 8,
        alternate: bool = True,
    ):
        fmaps, *_ = buffer.get_neural_attrs()
        self._altcorr.build_pyramid(fmaps[: len(buffer), 0].float())

        for step in range(steps):
            poses, disps, intrinsics = buffer.get_geometric_attrs()
            ii, jj = graph.get_edges()
            coords1, _ = projective_transform(poses, disps, intrinsics, ii, jj)

            target, _ = graph.get_flow_attrs()
            motn = (
                torch.cat([coords1 - self._coords0, target - coords1], dim=-1)
                .permute(0, 3, 1, 2)
                .clamp(-64.0, 64.0)
            )

            corr1 = self._altcorr(coords1.float(), ii, jj + (ii == jj).long())
            net_hidden = graph.get_neural_attrs()
            _, inps, _ = buffer.get_neural_attrs()
            inps_ii = inps[ii].half()
            ii, jj = graph.get_edges()

            net, delta, weight, damping, upmask = self._droid.apply_update(
                net_hidden,
                inps_ii,
                corr1,
                motn,
                ii,
                jj,
            )

            target = coords1 + delta.float()
            weight = weight.float()
            unique_ii = torch.unique(ii)

            graph.update_damping(unique_ii, damping)

            # BA
            damping = graph.get_damping(unique_ii)
            eta = 0.2 * damping + 1e-7

            ii, jj = graph.get_edges()
            prev_target, prev_weight = graph.get_flow_attrs()
            ctx, params = self._get_ba_params(
                buffer=buffer,
                step=step,
                alternate=alternate,
                ii=ii,
                jj=jj,
                target=prev_target,
                weight=prev_weight,
                eta=eta,
                t0=t0,
                lm=1e-5,
                ep=1e-2,
            )
            if params is not None:
                result = self._dspo(ctx, params)
                buffer.apply_ba_result(result)

            buffer.upsample_disps(unique_ii, upmask)
            graph.update_residuals(target, weight, net)

        self._altcorr.clear()  # gc pyramid

    def _get_ba_params(
        self,
        buffer: KeyFrameBuffer,
        step: int,
        alternate: bool,
        ii: torch.Tensor,
        jj: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        eta: torch.Tensor,
        t0: int,
        lm: float,
        ep: float,
    ) -> Union[Tuple[BAContext, dict] | Tuple[None, None]]:
        t0 = max(1, t0)

        if not alternate or step % 2 == 0:
            ctx = buffer.create_ba_context(BAType.POSE_DEPTH)
            params = {
                "target": target,
                "weight": weight,
                "eta": eta,
                "ii": ii,
                "jj": jj,
                "num_fixed_poses": t0,
                "lm": lm,
                "ep": ep,
                "iters": 2,
                "alpha": 0.05,
            }
            return ctx, params
        else:
            ctx = buffer.create_ba_context(BAType.DEPTH_SCALE)
            ii_f, jj_f, tgt_f, wt_f, eta_f = self._filter_mono_edges(
                ctx.invalid_mono_frames, ii, jj, target, weight, eta
            )
            if ii_f.shape[0] == 0:
                return None, None
            params = {
                "target": tgt_f,
                "weight": wt_f,
                "eta": eta_f,
                "ii": ii_f,
                "jj": jj_f,
                "num_fixed_poses": t0,
                "lm": lm,
                "ep": ep,
                "iters": 2,
                "alpha": 0.01,
            }
            return ctx, params

    def _filter_mono_edges(self, invalid_mono_frames, ii, jj, target, weight, eta):
        if invalid_mono_frames is None:
            return ii, jj, target, weight, eta

        invalid_idx = torch.where(invalid_mono_frames)[0]
        if invalid_idx.numel() == 0:
            return ii, jj, target, weight, eta

        mask = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        for idx in invalid_idx:
            mask = mask | (ii == idx) | (jj == idx)

        keep = ~mask
        ii_f, jj_f = ii[keep], jj[keep]

        # eta is indexed by unique(ii), so filter to match
        orig_unique = torch.unique(ii)
        new_unique = torch.unique(ii_f)
        valid = torch.tensor([u in new_unique for u in orig_unique]).to(ii.device)

        return ii_f, jj_f, target[keep], weight[keep], eta[valid]

    def __call__(
        self,
        buffer: KeyFrameBuffer,
        graph: FactorGraph,
        steps: int = 1,
        alternate: bool = True,
        t0: Optional[int] = None,
    ) -> None:
        if isinstance(graph, LocalGraph):
            self._local_update(
                buffer=buffer, graph=graph, steps=steps, alternate=alternate, t0=t0
            )
        elif isinstance(graph, GlobalGraph):
            self._global_update(
                buffer=buffer, graph=graph, t0=t0, steps=steps, alternate=alternate
            )
