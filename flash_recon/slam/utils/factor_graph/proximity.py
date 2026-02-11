from copy import deepcopy

import numpy as np
import torch


def _suppress_nms(d, i, j, nms, t0, t1, t, stride):
    """Zero out nearby entries in distance matrix around (i, j)."""
    radius = max(min(abs(i - j) - 2, nms), 0)
    for di in range(-nms, nms + 1):
        for dj in range(-nms, nms + 1):
            if abs(di) + abs(dj) <= radius:
                i1, j1 = i + di, j + dj
                if t0 <= i1 < t and t1 <= j1 < t:
                    d[(i1 - t0) * stride + (j1 - t1)] = np.inf


def find_local_edges(
    distance_fn,
    existing_edges,
    t0,
    t1,
    t,
    *,
    rad=2,
    nms=2,
    beta=0.25,
    thresh=16.0,
    max_factors=75,
    device="cuda",
):
    """local proximity NMS — select edges between nearby frames.

    Args:
        distance_fn: (ii, jj, beta=) -> distances
        existing_edges: (ii, jj) of all current edges (active + bad + inactive)
        t0: start of source frame range
        t1: start of target frame range
        t: total keyframe count
        rad, nms, beta, thresh, max_factors: NMS parameters

    Returns:
        (ii, jj) tensor pair on `device`.
    """
    ix = torch.arange(t0, t)
    jx = torch.arange(t1, t)
    ii, jj = torch.meshgrid(ix, jx, indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)
    stride = t - t1

    dist = distance_fn(ii, jj, beta=beta)
    dist[ii - rad < jj] = np.inf
    dist[dist > 100] = np.inf

    # Suppress around existing edges
    ex_ii, ex_jj = existing_edges
    for i, j in zip(ex_ii.cpu().numpy(), ex_jj.cpu().numpy()):
        _suppress_nms(dist, i, j, nms, t0, t1, t, stride)

    # Seed with direct temporal neighbors
    edges = []
    for i in range(t0, t):
        for j in range(max(i - rad - 1, 0), i):
            edges.append((i, j))
            edges.append((j, i))
            dist[(i - t0) * stride + (j - t1)] = np.inf

    # Greedily add closest remaining pairs with NMS
    order = torch.argsort(dist)
    for k in order:
        if dist[k].item() > thresh:
            continue
        if len(edges) > max_factors:
            break
        i, j = ii[k], jj[k]
        edges.append((i, j))
        edges.append((j, i))
        _suppress_nms(dist, i, j, nms, t0, t1, t, stride)

    return torch.as_tensor(edges, device=device).unbind(dim=-1)


def find_global_edges(
    distance_fn,
    t_start,
    t_end,
    *,
    t_start_loop=None,
    nms=12,
    radius=1,
    thresh=25.0,
    max_factors=200,
    beta=0.75,
    loop=True,
):
    """global proximity NMS — select edges for loop closure.

    Args:
        distance_fn: (ii, jj, beta=) -> distances
        t_start: start of target frame range
        t_end: end of frame range
        t_start_loop: start of source frame range (defaults to t_start)
        nms, radius, thresh, max_factors, beta: NMS parameters
        loop: expand neighbors for loop-closure edges

    Returns:
        (ii, jj) on CPU, or None if too few edges found.
    """
    if t_start_loop is None or not loop:
        t_start_loop = t_start
    assert t_start_loop >= t_start

    n_src = t_end - t_start_loop
    n_tgt = t_end - t_start
    ix = torch.arange(t_start_loop, t_end)
    jx = torch.arange(t_start, t_end)
    ii, jj = torch.meshgrid(ix, jx, indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)

    dist = distance_fn(ii, jj, beta=beta)
    raw_dist = deepcopy(dist).reshape(n_src, n_tgt)
    dist[ii - radius < jj] = np.inf
    dist[dist > thresh] = np.inf
    dist = dist.reshape(n_src, n_tgt)

    # Seed with direct temporal neighbors
    edges = []
    for i in range(t_start_loop, t_end):
        for j in range(max(i - radius - 1, 0), i):
            edges.append((i, j))
            edges.append((j, i))
            dist[i - t_start_loop, j - t_start] = np.inf

    vals, order = torch.sort(dist.reshape(-1), descending=False)
    order = order[vals <= thresh].tolist()

    loop_edges = 0
    for k in order:
        di, dj = k // n_tgt, k % n_tgt
        if dist[di, dj].item() > thresh:
            continue
        if len(edges) > max_factors:
            break

        i, j = ii[k], jj[k]

        if loop:
            # Expand to neighboring frame pairs
            sub = []
            for si in range(max(i - 1, t_start_loop), min(i + 2, t_end)):
                for sj in range(max(j - 1, t_start), min(j + 2, t_end)):
                    if raw_dist[si - t_start_loop, sj - t_start] <= thresh:
                        if si != sj and si - sj > 20:
                            sub.append((si, sj))
            edges += sub
            loop_edges += len(sub)
        else:
            edges += [(i, j), (j, i)]

        dist[
            max(0, di - nms) : min(n_src, di + nms + 1),
            max(0, dj - nms) : min(n_tgt, dj + nms + 1),
        ] = np.inf

    if len(edges) < 3 or (loop and loop_edges == 0):
        return None

    return torch.tensor(edges, device="cpu").unbind(dim=-1)
