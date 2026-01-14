import sys
import time
import torch
import lietorch

for p in ["/workspace", "/workspace/Splat-SLAM"]:
    if p not in sys.path:
        sys.path.append(p)

from originals import add_frontend_proximity_factors as old_add_frontend_proximity_factors
from new import add_frontend_proximity_factors as new_add_frontend_proximity_factors


def run_case(T, max_factors, t0, t1, rad=2, nms=2, thresh=16.0, iters=50, seed=123, device="cpu"):
    torch.manual_seed(seed)
    device = torch.device(device)

    # Edges
    if T < 2:
        ii = torch.zeros(0, device=device, dtype=torch.long)
        jj = torch.zeros(0, device=device, dtype=torch.long)
        ii_bad = torch.zeros(0, device=device, dtype=torch.long)
        jj_bad = torch.zeros(0, device=device, dtype=torch.long)
        ii_inac = torch.zeros(0, device=device, dtype=torch.long)
        jj_inac = torch.zeros(0, device=device, dtype=torch.long)
    else:
        ii = torch.arange(0, T - 1, device=device, dtype=torch.long)
        jj = torch.arange(1, T, device=device, dtype=torch.long)
        ii_bad = torch.arange(0, T - 1, device=device, dtype=torch.long)
        jj_bad = torch.arange(1, T, device=device, dtype=torch.long)
        ii_inac = torch.arange(0, T - 1, device=device, dtype=torch.long)
        jj_inac = torch.arange(1, T, device=device, dtype=torch.long)

    # Distance vector for [t0..T) x [t1..T)
    L = max(T - t0, 0) * max(T - t1, 0)
    d = 50.0 * torch.rand(L, device=device)



    with torch.inference_mode():
        for _ in range(10):
            old_add_frontend_proximity_factors(
                T,
                d.clone(),
                ii,
                ii_bad,
                ii_inac,
                jj,
                jj_bad,
                jj_inac,
                max_factors,
                t0,
                t1,
                rad,
                nms,
                thresh,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            old_ii, old_jj = old_add_frontend_proximity_factors(
                T,
                d.clone(),
                ii,
                ii_bad,
                ii_inac,
                jj,
                jj_bad,
                jj_inac,
                max_factors,
                t0,
                t1,
                rad,
                nms,
                thresh,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        old_latency = (time.perf_counter() - start) * 1000.0 / iters

    with torch.inference_mode():
        for _ in range(10):
            new_add_frontend_proximity_factors(
                d.clone(),
                ii,
                ii_bad,
                ii_inac,
                jj,
                jj_bad,
                jj_inac,
                max_factors,
                T,
                t0,
                t1,
                rad,
                nms,
                thresh,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            new_ii, new_jj = new_add_frontend_proximity_factors(
                d.clone(),
                ii,
                ii_bad,
                ii_inac,
                jj,
                jj_bad,
                jj_inac,
                max_factors,
                T,
                t0,
                t1,
                rad,
                nms,
                thresh,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        new_latency = (time.perf_counter() - start) * 1000.0 / iters

    ii_match_rate = (new_ii == old_ii).float().mean().item()
    jj_match_rate = (new_jj == old_jj).float().mean().item()

    stats = {
        "ii Match Rate": ii_match_rate,
        "jj Matchc Rate": jj_match_rate,
        "Old Latency (ms)": old_latency,
        "New Latency (ms)": new_latency,
        "Speedup x": old_latency / new_latency,
    }

    return stats


if __name__ == "__main__":
    import pprint

    cases = [
        dict(T=3, max_factors=64, t0=0, t1=0, rad=2, nms=2, thresh=16.0, iters=200, seed=1),
        dict(T=4, max_factors=128, t0=0, t1=0, rad=2, nms=2, thresh=16.0, iters=200, seed=2),
        dict(T=8, max_factors=256, t0=0, t1=0, rad=2, nms=2, thresh=16.0, iters=100, seed=3),
    ]

    for i, cfg in enumerate(cases):
        print(f"Case {i+1}:")
        print("  Config:")
        pprint.pprint(cfg, indent=4)
        stats = run_case(**cfg)
        print("  Stats:")
        pprint.pprint(stats, indent=4, sort_dicts=False)
        print()
