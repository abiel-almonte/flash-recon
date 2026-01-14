import torch
from geometry import get_meshgrid

def add_frontend_proximity_factors(
    dist: torch.Tensor,
    self_ii,
    self_ii_bad,
    self_ii_inac,
    self_jj,
    self_jj_bad,
    self_jj_inac,
    max_factors,
    count,
    t0: int = 0,
    t1: int = 0,
    rad: int = 2,
    nms: int = 2,
    thresh: float = 16.0,
):
    """Add proximity-based edges"""

    gpu_device = "cuda"
    cpu_device = torch.device("cpu")
    stride = count - t1

    ix = torch.arange(t0, count)
    jx = torch.arange(t1, count)

    ii, jj = torch.meshgrid(ix, jx,indexing="ij")
    ii = ii.flatten()
    jj = jj.flatten()

    d = dist.detach().to(cpu_device)
    d[(ii - rad) < jj] = torch.inf
    d[d > 100] = torch.inf

    ii1: torch.Tensor = torch.cat([self_ii, self_ii_bad, self_ii_inac], dim=0)
    jj1: torch.Tensor = torch.cat([self_jj, self_jj_bad, self_jj_inac], dim=0)

    for i, j in zip(ii1.tolist(), jj1.tolist()):
        r = max(min(abs(i - j) - 2, nms), 0)

        for di in range(-nms, nms+1):

            for dj in range(-nms, nms+1):
                if abs(di) + abs(dj) <= r:
                    i1 = i + di
                    j1 = j + dj

                    if (t0 <= i1 < count) and (t1 <= j1 < count):
                        flat_idx = (i1-t0)*stride + (j1-t1)
                        d[flat_idx] = torch.inf

    es = []
    for i in range(t0, count):
        for j in range(max(i-rad-1,0), i):
            es.append((i,j))
            es.append((j,i))

            flat_idx = (i - t0) * stride + (j - t1)
            d[flat_idx] = torch.inf

    order = torch.argsort(d)
    cap = max_factors if max_factors > 0 else float("inf")
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

    ii_new, jj_new = torch.as_tensor(
        es, device=gpu_device, dtype=torch.long
    ).unbind(dim=-1)

    return ii_new, jj_new