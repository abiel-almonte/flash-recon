import torch
from copy import deepcopy

from geometry import projective_transform, compute_distance

from .. import KeyFrameBuffer

from .base import FactorGraph
from .proximity import find_global_edges


class GlobalGraph(FactorGraph):

    @classmethod
    def from_local(cls, src):
        g = cls.__new__(cls)
        g.cfg = src.cfg
        g.ht = src.ht
        g.wd = src.wd
        g.device = src.device

        g.ii = deepcopy(src.ii)
        g.jj = deepcopy(src.jj)
        g.age = deepcopy(src.age)
        g.net_hidden = deepcopy(src.net_hidden)
        g.damping = 1e-6 * torch.ones_like(src.damping)
        g.target = deepcopy(src.target)
        g.weight = deepcopy(src.weight)

        return g

    @torch.no_grad()
    def add(self, ii, jj, net, target, *, max_factors, remove=False):
        ii, jj = self._to_long(ii).contiguous(), self._to_long(jj).contiguous()

        keep = self._dedup_mask(ii, jj)
        ii, jj = ii[keep].contiguous(), jj[keep].contiguous()
        net = net[keep].contiguous()
        target = target[keep].contiguous()

        if ii.shape[0] == 0:
            return

        left = max_factors - len(self)
        if left > 0 and remove and self.ii.shape[0] + ii.shape[0] > max_factors:
            self._evict_oldest(ii.shape[0], max_factors)

        self._append(ii, jj, net, target, torch.zeros_like(target))

    def add_proximity_edges(
        self,
        buffer: KeyFrameBuffer,
        t_start: int,
        t_end: int,
        t_start_loop: int,
        nms: int,
        radius: int,
        thresh: float,
        max_factors: int,
        beta: float,
        loop=True,
    ):
        """Find loop-closure edges via NMS, look up features, and add."""

        count = len(buffer)
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

        edges = find_global_edges(
            distance_fn,
            t_start,
            t_end,
            t_start_loop=t_start_loop,
            nms=nms,
            radius=radius,
            thresh=thresh,
            max_factors=max_factors,
            beta=beta,
            loop=loop,
        )
        if edges is None:
            return 0

        ii = edges[0].to(self.device)
        jj = edges[1].to(self.device)

        poses, disps, intrinsics = buffer.get_geometric_attrs()
        _, nets, _ = buffer.get_neural_attrs()
        nets = nets[ii].half()

        target, _ = projective_transform(poses, disps, intrinsics, ii, jj)

        self.add(ii, jj, nets, target, max_factors=self.max_factors, remove=True)
        return len(ii)
