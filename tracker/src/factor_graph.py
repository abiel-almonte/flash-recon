import torch
import numpy as np

from geometry import get_meshgrid, projective_transform
from neural import CorrBlock


class FactorGraph:
    def __init__(self, cfg):
        self.device = cfg.get("device", "cuda")
        self.down_scale = int(cfg.get("cam", {}).get("down_scale", 8))
        H_out = int(cfg.get("cam", {}).get("H_out", 480))
        W_out = int(cfg.get("cam", {}).get("W_out", 640))
        ht = H_out // self.down_scale
        wd = W_out // self.down_scale

        buffer_capacity = int(cfg.get("tracking", {}).get("buffer", 512))
        self.max_factors = int(cfg.get("tracking", {}).get("max_factors", -1))
        self.radius = int(cfg.get("tracking", {}).get("frontend", {}).get("radius", 3))
        self.coords0 = torch.stack(
            **get_meshgrid(ht, wd, device=self.device, dtype=torch.float), dim=-1
        )
        self.ii = torch.as_tensor([], device=self.device, dtype=torch.long)
        self.jj = torch.as_tensor([], device=self.device, dtype=torch.long)
        self.age = torch.as_tensor([], device=self.device, dtype=torch.long)
        self.damping = torch.ones(buffer_capacity, ht, wd, device=self.device) * 1e-6
        self.target = torch.zeros([0, ht, wd, 2], device=self.device)
        self.weight = torch.zeros([0, ht, wd, 2], device=self.device)

        self.ii_inac = torch.as_tensor([], device=self.device, dtype=torch.long)
        self.jj_inac = torch.as_tensor([], device=self.device, dtype=torch.long)
        self.ii_bad = torch.as_tensor([], device=self.device, dtype=torch.long)
        self.jj_bad = torch.as_tensor([], device=self.device, dtype=torch.long)
        self.target_inac = torch.zeros([0, ht, wd, 2], device=self.device)
        self.weight_inac = torch.zeros([0, ht, wd, 2], device=self.device)

        self.corr = CorrBlock()
        self.net = None
        self.inp = None

    def _remove_duplicates(self, ii, jj):
        """remove duplicate edges"""
        curr_ii = torch.cat([self.ii, self.ii_inac], dim=0)
        curr_jj = torch.cat([self.ii, self.ii_inac], dim=0)

        encoding_stride = max(curr_ii.max(), curr_jj.max(), ii.max(), jj.max())

        edges = ii * encoding_stride + jj
        curr_edges = curr_ii * encoding_stride + curr_jj

        keep = torch.isin(edges, curr_edges, invert=True)

        return ii[keep], jj[keep]

    def remove_factors(self, remove, store_inac=False):
        """drop edges from factor graph"""

        if store_inac:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[remove]], dim=0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[remove]], dim=0)
            self.target_inac = torch.cat([self.target_inac, self.target[remove]], dim=0)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[remove]], dim=0)

        keep = ~remove

        self.ii = self.ii[keep]
        self.jj = self.jj[keep]
        self.age = self.age[keep]
        self.target = self.target[keep]
        self.weight = self.weight[keep]

        if self.corr is not None:
            self.corr = self.corr[keep]

        if self.net is not None:
            self.net = self.net[:, keep]

        if self.inp is not None:
            self.inp = self.inp[:, keep]

    def remove_keyframe(self, ix):
        """drop edges from factor graph"""

        m = (self.ii_inac == ix) | (self.jj_inac == ix)

        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            inv_m = ~m
            self.ii_inac = self.ii_inac[inv_m]
            self.jj_inac = self.jj_inac[inv_m]
            self.target_inac = self.target_inac[inv_m]
            self.weight_inac = self.weight_inac[inv_m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.remove_factors(m, store=False)

    def add_factors(self, ii, jj, buffer_payload, remove=False):
        """add edges to factor graph"""

        ii, jj = self._remove_duplicates(ii, jj)
        if ii.size(0) == 0:
            return

        if (
            self.max_factors > 0
            and self.ii.size(0) + ii.size(0) > self.max_factors
            and self.corr is not None
            and remove
        ):
            ix_sorted = torch.argsort(self.age)
            num_to_remove = self.age.size(0) - self.max_factors + ii.size(0)

            if num_to_remove > 0:
                remove_mask = torch.zeros_like(self.age, dtype=torch.bool)
                remove_mask[ix_sorted[-num_to_remove:]] = True
                self.remove_factors(remove_mask, store_inac=True)

        target, _ = projective_transform(
            buffer_payload.poses,
            buffer_payload.disps,
            buffer_payload.intrinsics,
            ii,
            jj,
            jacobian=False,
        )
        weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], dim=0)
        self.jj = torch.cat([self.jj, jj], dim=0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], dim=0)
        self.target = torch.cat([self.target, target], dim=0)
        self.weight = torch.cat([self.weight, weight], dim=0)

        fmap1 = buffer_payload.fmaps[ii, 0]
        fmap2 = buffer_payload.fmaps[jj, ii == jj]
        self.corr.build_pyramid(fmap1, fmap2)

        net = buffer_payload.nets[ii]
        if self.net is None:
            self.net = net
        else:
            torch.cat([self.net, net], dim=1)

        inp = buffer_payload.inps[ii]
        if self.inp is None:
            self.inp = inp
        else:
            torch.cat([self.inp, inp], dim=1)

    def add_neighborhood_factors(self, t0, t1, buffer_payload):
        """add edges between neighboring frames within radius"""

        ii, jj = get_meshgrid((t0, t1), (t0, t1), device=self.device, dtype=torch.long)
        ii = ii.flatten()
        jj = jj.flatten()

        keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= self.radius)

        self.add_factors(ii[keep], jj[keep], buffer_payload)

    def add_frontend_proximity_factors(
        self,
        count: int,
        dist: torch.Tensor,
        buffer_payload,
        t0: int = 0,
        t1: int = 0,
        rad: int = 2,
        nms: int = 2,
        thresh: float = 16.0,
        remove: bool = False,
    ):
        """Add proximity-based edges"""

        gpu_device = self.device
        cpu_device = torch.device("cpu")
        stride = count - t1

        ii, jj = get_meshgrid((t0, count), (t1, count), device=cpu_device, dtype=torch.long)
        ii = ii.flatten()
        jj = jj.flatten()

        d = dist.detach().to(cpu_device)
        d[(ii - rad) < jj] = torch.inf
        d[d > 100] = torch.inf

        es = []
        for i in range(t0, count):
            j_start = max(i - rad - 1, t1)
            for j in range(j_start, i):
                es.append((i, j))
                es.append((j, i))
                flat_idx = (i - t0) * stride + (j - t1)
                d[flat_idx] = torch.inf

        order = torch.argsort(d)
        cap = self.max_factors if self.max_factors > 0 else float("inf")

        for k in order.tolist():
            if d[k].item() > thresh:
                continue
            if len(es) > cap:
                break

            i = int(ii[k].item())
            j = int(jj[k].item())

            es.append((i, j))
            es.append((j, i))

            r = max(min(abs(i - j) - 2, nms), 0)
            for di in range(-nms, nms + 1):
                for dj in range(-nms, nms + 1):
                    if abs(di) + abs(dj) <= r:
                        i1 = i + di
                        j1 = j + dj
                        if (t0 <= i1 < count) and (t1 <= j1 < count):
                            flat_idx = (i1 - t0) * stride + (j1 - t1)
                            d[flat_idx] = torch.inf

        if len(es) < 1:
            return

        ii_new, jj_new = torch.as_tensor(es, device=cpu_device, dtype=torch.long).unbind(dim=-1)
        ii_new = ii_new.to(gpu_device)
        jj_new = jj_new.to(gpu_device)
        self.add_factors(ii_new, jj_new, buffer_payload, remove)

    def add_backend_proximity_factors(
        self,
        dist: torch.Tensor,
        buffer_payload,
        t0: int = 0,
        t1: int = 0,
        rad: int = 2,
        nms: int = 2,
        thresh: float = 16.0,
        max_factors: int = 500,
        t0_loop=None,
        loop=False,
    ):
        if t0_loop is None or not loop:
            t0_loop = t0
        assert t0_loop >= t0, f"short: {t0_loop}, long: {t0}."

        gpu_device = self.device
        cpu_device = torch.device("cpu")

        ilen = t1 - t0_loop
        jlen = t1 - t0

        ix = torch.arange(t0_loop, t1, device=cpu_device)
        jx = torch.arange(t0, t1, device=cpu_device)

        ii, jj = torch.meshgrid(ix, jx, indexing="ij")
        ii = ii.flatten()
        jj = jj.flatten()

        d = dist.detach().to(cpu_device)
        rawd = d.clone().reshape(ilen, jlen)
        d[(ii - rad) < jj] = torch.inf
        d[d > thresh] = torch.inf
        d = d.reshape(ilen, jlen)

        edges = []
        for i in range(t0_loop, t1):
            for j in range(max(i - rad - 1, t0), i):
                edges.append((i, j))
                edges.append((j, i))
                di = i - t0_loop
                dj = j - t0
                d[di, dj] = torch.inf

        vals, flat_ix = torch.sort(d.reshape(-1), descending=False)
        flat_ix = flat_ix[vals <= thresh]

        loop_edges = 0
        n_neighboring = 1
        cap = max_factors if max_factors > 0 else float("inf")

        if len(edges) <= cap:
            iix = ii[flat_ix]
            jjx = jj[flat_ix]

            if loop:
                window = torch.arange(-n_neighboring, n_neighboring + 1, device=cpu_device)
                si = (window + iix[:, None]).clamp(t0_loop, t1 - 1)
                sj = (window + jjx[:, None]).clamp(t0, t1 - 1)
                SI, SJ = torch.meshgrid(si.flatten(), sj.flatten(), indexing="ij")

                mask = (rawd[(SI - t0_loop).long(), (SJ - t0).long()] <= thresh) & (SI - SJ > 20)

                sub_edges = torch.stack([SI[mask], SJ[mask]], dim=-1)
                loop_edges += sub_edges.size(0)
                if sub_edges.numel() > 0:
                    edges += [(int(a), int(b)) for a, b in sub_edges.tolist()]
            else:
                edges += [(int(a), int(b)) for a, b in torch.stack([iix, jjx], dim=-1).tolist()]
                edges += [(int(a), int(b)) for a, b in torch.stack([jjx, iix], dim=-1).tolist()]

            di = (flat_ix // jlen)
            dj = (flat_ix % jlen)

            offsets = torch.arange(-nms, nms + 1, device=cpu_device)
            dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
            dy = dy.reshape(-1)
            dx = dx.reshape(-1)

            yi = (di[:, None] + dy[None, :]).clamp(0, ilen - 1)
            xj = (dj[:, None] + dx[None, :]).clamp(0, jlen - 1)

            flat_idx = (yi.reshape(-1) * jlen + xj.reshape(-1)).long()
            d = d.reshape(-1)
            d[flat_idx] = torch.inf
            d = d.reshape(ilen, jlen)

        if len(edges) < 3 or (loop and loop_edges == 0):
            return 0

        ii_new, jj_new = torch.as_tensor(edges, device=cpu_device, dtype=torch.long).unbind(dim=-1)
        ii_new = ii_new.to(gpu_device)
        jj_new = jj_new.to(gpu_device)
        self.add_factors(ii_new, jj_new, buffer_payload, remove=True)

        return len(self.ii)

    def clear(self):
        self.ii = None
        self.jj = None
        self.age = None
        self.damping = None
        self.target = None
        self.weight = None

        self.ii_inac = None
        self.jj_inac = None
        self.ii_bad = None
        self.jj_bad = None
        self.target_inac = None
        self.weight_inac = None

        self.corr = None
        self.net = None
        self.inp = None
