import numpy as np
import torch

def add_frontend_proximity_factors(
    t,
    d,
    self_ii,
    self_ii_bad,
    self_ii_inac,
    self_jj,
    self_jj_bad,
    self_jj_inac,
    max_factors,
    t0=0,
    t1=0,
    rad=2,
    nms=2,
    thresh=16.0,
):

    ix = torch.arange(t0, t)
    jx = torch.arange(t1, t)

    ii, jj = torch.meshgrid(ix, jx,indexing="ij")
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)

    d[ii - rad < jj] = np.inf
    d[d > 100] = np.inf

    ii1 = torch.cat([self_ii, self_ii_bad, self_ii_inac], 0)
    jj1 = torch.cat([self_jj, self_jj_bad, self_jj_inac], 0)
    for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
        for di in range(-nms, nms+1):
            for dj in range(-nms, nms+1):
                if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                    i1 = i + di
                    j1 = j + dj

                    if (t0 <= i1 < t) and (t1 <= j1 < t):
                        d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf


    es = []
    for i in range(t0, t):
        for j in range(max(i-rad-1,0), i):
            es.append((i,j))
            es.append((j,i))
            d[(i-t0)*(t-t1) + (j-t1)] = np.inf

    ix = torch.argsort(d)
    for k in ix:
        if d[k].item() > thresh:
            continue

        if len(es) > max_factors:
            break

        i = ii[k]
        j = jj[k]
        
        es.append((i, j))
        es.append((j, i))

        for di in range(-nms, nms+1):
            for dj in range(-nms, nms+1):
                if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                    i1 = i + di
                    j1 = j + dj

                    if (t0 <= i1 < t) and (t1 <= j1 < t):
                        d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

    ii, jj = torch.as_tensor(es, device="cuda").unbind(dim=-1)
    return ii, jj

