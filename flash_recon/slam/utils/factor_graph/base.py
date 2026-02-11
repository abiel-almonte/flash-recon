from typing import Optional, Tuple
import torch

Tensors = Tuple[torch.Tensor, ...]


class FactorGraph:
    def __init__(self, cfg):
        self.cfg = cfg

        device = cfg.get("device", "cuda")
        down = int(cfg["cam"].get("down_scale", 8))
        ht = int(cfg["cam"]["H_out"]) // down
        wd = int(cfg["cam"]["W_out"]) // down
        capacity = int(cfg["tracking"]["buffer"])

        self.device = device
        self.down_scale = down
        self.ht = ht
        self.wd = wd
        self.max_factors = int(cfg["tracking"]["local"]["max_factors"])

        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)
        self.net_hidden = None
        self.damping = 1e-6 * torch.ones(capacity, ht, wd, device=device)
        self.target = torch.zeros([0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([0, ht, wd, 2], device=device, dtype=torch.float)

    def __len__(self) -> int:
        return self.ii.shape[0]

    def _to_long(self, t) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            return torch.as_tensor(t, dtype=torch.long, device=self.device)
        return t.contiguous()

    def _dedup_mask(self, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
        existing = self._all_edge_set()
        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in existing
        return keep

    def get_neural_attrs(self) -> torch.Tensor:
        return self.net_hidden

    def get_flow_attrs(self) -> Tensors:
        return self.target, self.weight

    def get_edges(self) -> Tensors:
        return (self.ii, self.jj)

    def age_edges(self) -> None:
        self.age += 1

    def update_damping(self, ii, damping) -> None:
        self.damping[ii] = damping

    def get_damping(self, ii: Optional[torch.Tensor]) -> torch.Tensor:
        if ii is not None:
            damping = self.damping[ii]
        else:
            damping = self.damping

        return damping.contiguous()

    def update_residuals(
        self, target: torch.Tensor, weight: torch.Tensor, net: torch.Tensor
    ) -> None:
        self.target = target
        self.weight = weight
        self.net_hidden = net

    def _all_edge_set(self) -> set:
        return set([(i.item(), j.item()) for i, j in zip(self.ii, self.jj)])

    def _evict_oldest(self, n_incoming: int, max_factors: int) -> None:
        ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
        self.remove(ix >= max_factors - n_incoming, store=True)

    def _append(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        net: Optional[torch.Tensor],
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)
        self.net_hidden = (
            net if self.net_hidden is None else torch.cat([self.net_hidden, net], 0)
        )
        self.target = torch.cat([self.target, target], 0)
        self.weight = torch.cat([self.weight, weight], 0)

    def remove(self, mask: torch.Tensor, store: bool = False) -> None:
        keep = ~mask
        self.ii = self.ii[keep]
        self.jj = self.jj[keep]
        self.age = self.age[keep]
        if self.net_hidden is not None:
            self.net_hidden = self.net_hidden[keep]
        self.target = self.target[keep]
        self.weight = self.weight[keep]

    def neighborhood_pairs(self, t0: int, t1: int, r: int = 3) -> Tensors:
        ii, jj = torch.meshgrid(
            torch.arange(t0, t1), torch.arange(t0, t1), indexing="ij"
        )
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)
        keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
        return ii[keep], jj[keep]
