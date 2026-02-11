from typing import Tuple
import torch

from neural import Corr
from geometry import projective_transform, compute_distance

from .base import FactorGraph, Tensors
from .proximity import find_local_edges
from .. import KeyFrameBuffer


class LocalGraph(FactorGraph):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.corr = None
        self.inp = None

        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.target_inac = torch.zeros(
            [0, self.ht, self.wd, 2], device=self.device, dtype=torch.float
        )
        self.weight_inac = torch.zeros(
            [0, self.ht, self.wd, 2], device=self.device, dtype=torch.float
        )

        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=self.device)

    @property
    def inactives(self) -> Tensors:
        return (self.ii_inac, self.jj_inac, self.target_inac, self.weight_inac)

    def get_neural_attrs(self) -> Tensors:
        return (super().get_neural_attrs(), self.inp)

    def corr_initialized(self) -> bool:
        return self.corr is not None

    def apply_corr(self, coords: torch.Tensor) -> torch.Tensor | None:
        if self.corr_initialized():
            return self.corr(coords)

    def _all_edge_set(self) -> tuple:
        return set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)]
            + [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)]
        )

    def merge_inactive(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        t0: int,
    ) -> Tensors:
        """Merge nearby inactive edges with active for BA."""
        m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
        return (
            torch.cat([self.ii_inac[m], ii], 0),
            torch.cat([self.jj_inac[m], jj], 0),
            torch.cat([self.target_inac[m], target], 0),
            torch.cat([self.weight_inac[m], weight], 0),
        )

    def existing_edges(self) -> Tensors:
        """All edges (active + bad + inactive) for proximity NMS."""
        return (
            torch.cat([self.ii, self.ii_bad, self.ii_inac], 0),
            torch.cat([self.jj, self.jj_bad, self.jj_inac], 0),
        )

    def add(self, ii, jj, net, inp, target, *, fmaps=None, remove=False) -> None:
        ii, jj = self._to_long(ii), self._to_long(jj)

        keep = self._dedup_mask(ii, jj)
        ii, jj = ii[keep].contiguous(), jj[keep].contiguous()
        net = net[keep].contiguous()
        inp = inp[keep].contiguous() if inp is not None else None
        target = target[keep].contiguous()
        if fmaps is not None:
            fmaps = (fmaps[0][keep].contiguous(), fmaps[1][keep].contiguous())

        if ii.shape[0] == 0:
            return

        if self.max_factors > 0 and self.corr is not None and remove:
            if self.ii.shape[0] + ii.shape[0] > self.max_factors:
                self._evict_oldest(ii.shape[0], self.max_factors)

        if fmaps is not None:
            fmap1, fmap2 = fmaps
            if self.corr is None:
                self.corr = Corr(self.cfg)
            self.corr.build_pyramid(fmap1, fmap2)

        self._append(ii, jj, net, target, torch.zeros_like(target))
        if inp is not None:
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 0)

    def add_from_buffer(
        self,
        buffer: KeyFrameBuffer,
        ii: torch.Tensor,
        jj: torch.Tensor,
        remove: bool = False,
    ) -> None:
        ii = self._to_long(ii)
        jj = self._to_long(jj)

        fmaps, nets, inps = buffer.get_neural_attrs()

        net = nets[ii].half()
        inp = inps[ii].half()
        c = (ii == jj).long()
        fmap1 = fmaps[ii, 0].half()
        fmap2 = fmaps[jj, c].half()

        poses, disps, intrinsics = buffer.get_geometric_attrs()
        target, _ = projective_transform(poses, disps, intrinsics, ii, jj)

        self.add(ii, jj, net, inp, target, fmaps=(fmap1, fmap2), remove=remove)

    def add_proximity_edges(
        self,
        buffer: KeyFrameBuffer,
        t0: int,
        t1: int,
        count: int,
        rad: int = 2,
        nms: int = 2,
        beta: float = 0.25,
        thresh: float = 16.0,
        max_factors: int = 75,
        remove: bool = False,
    ) -> None:
        """Find nearby edges via NMS and add from buffer."""

        poses, disps, intrinsics = buffer.get_geometric_attrs()
        poses = poses[:count]
        disps = disps[:count]

        distance_fn = lambda ii, jj, beta: (
            compute_distance(
                poses,
                disps,
                intrinsics,
                ii=self._to_long(ii).reshape(-1).to(self.device),
                jj=self._to_long(jj).reshape(-1).to(self.device),
                beta=beta,
                bidirectional=True,
            )
        )

        edges = find_local_edges(
            distance_fn,
            self.existing_edges(),
            t0,
            t1,
            count,
            rad=rad,
            nms=nms,
            beta=beta,
            thresh=thresh,
            max_factors=max_factors,
            device=self.device,
        )
        self.add_from_buffer(buffer, *edges, remove=remove)

    @torch.autocast("cuda", enabled=True)
    def remove(self, mask: torch.Tensor, store: bool = False) -> None:
        """Remove edges. If store=True, save as inactive first."""
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[mask]], 0)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[mask]], 0)

        keep = ~mask
        if self.corr is not None:
            self.corr.filter_pyramid(keep)
        if self.inp is not None:
            self.inp = self.inp[keep]

        super().remove(mask, store=False)

    @torch.autocast("cuda", enabled=True)
    def remove_keyframe(self, ix: int) -> None:
        # Shift + prune inactive edges
        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1
        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[~m]
            self.weight_inac = self.weight_inac[~m]

        # Shift + prune active edges
        m = (self.ii == ix) | (self.jj == ix)
        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.remove(m, store=False)
