#!/usr/bin/env python3

import sys
import torch
import lietorch

for p in ["/workspace"]:
    if p not in sys.path:
        sys.path.append(p)

from geometry import Intrinsics, matrix_to_pose
from tracker import FactorGraph, BufferPayload, ProximityPayload, CallerRole


def make_cfg(H=240, W=320):
    return {
        "device": "cuda",
        "cam": {
            "down_scale": 8,
            "H_out": H,
            "W_out": W,
            "fx": 300.0,
            "fy": 300.0,
            "cx": W / 2.0,
            "cy": H / 2.0,
        },
        "tracking": {
            "frontend": {"radius": 3},
            "buffer": 64,
            "max_factors": 100,
        },
    }


def make_buffer_payload(T, H, W, down_scale, device, dtype):
    """Create a mock BufferPayload for testing."""
    ht, wd = H // down_scale, W // down_scale
    
    # Random poses
    xi = torch.randn(T, 6, device=device, dtype=dtype) * 0.05
    poses_se3 = lietorch.SE3.exp(xi)
    poses = matrix_to_pose(poses_se3.matrix())
    
    # Random disps and features
    disps = torch.rand(T, ht, wd, device=device, dtype=dtype).clamp_min(1e-4)
    intrinsics = Intrinsics(300.0, 300.0, W / 2.0, H / 2.0, device=device)
    fmaps = torch.randn(T, 1, 128, ht, wd, device=device, dtype=dtype)
    nets = torch.randn(T, 128, ht, wd, device=device, dtype=dtype)
    inps = torch.randn(T, 128, ht, wd, device=device, dtype=dtype)
    
    return BufferPayload(
        count=T,
        poses=poses,
        disps=disps,
        intrinsics=intrinsics,
        fmaps=fmaps,
        nets=nets,
        inps=inps,
    )


@torch.inference_mode()
def test_initialization():
    """Test FactorGraph initialization."""
    cfg = make_cfg()
    graph = FactorGraph(cfg)
    
    assert graph.ii.numel() == 0
    assert graph.jj.numel() == 0
    assert graph.age.numel() == 0
    assert graph.target.size(0) == 0
    assert graph.weight.size(0) == 0
    assert graph.coords0.shape == (30, 40, 2)  # H//8, W//8, 2
    assert graph.corr is not None
    print("  ✓ Initialization")


@torch.inference_mode()
def test_add_factors():
    """Test adding factors to the graph."""
    cfg = make_cfg(H=96, W=128)
    graph = FactorGraph(cfg)
    device = torch.device("cuda")
    
    T = 5
    payload = make_buffer_payload(T, 96, 128, 8, device, torch.float32)
    
    # Add edges between frames 0-1 and 1-2
    ii = torch.tensor([0, 1], device=device, dtype=torch.long)
    jj = torch.tensor([1, 2], device=device, dtype=torch.long)
    
    graph.add_factors(ii, jj, payload)
    
    assert graph.ii.numel() == 2
    assert graph.jj.numel() == 2
    assert graph.target.size(0) == 2
    assert graph.weight.size(0) == 2
    assert graph.age.numel() == 2
    assert (graph.age == 0).all()
    assert graph.corr.pyramid is not None
    print("  ✓ Add factors")


@torch.inference_mode()
def test_add_neighborhood_factors():
    """Test adding neighborhood factors."""
    cfg = make_cfg(H=96, W=128)
    cfg["tracking"]["frontend"]["radius"] = 2
    graph = FactorGraph(cfg)
    device = torch.device("cuda")
    
    T = 5
    payload = make_buffer_payload(T, 96, 128, 8, device, torch.float32)
    
    # Add neighborhood factors for frames 0-4
    graph.add_neighborhood_factors(0, 4, payload)
    
    # With radius=2 and 4 frames, should have edges:
    # (0,1), (1,0), (0,2), (2,0), (1,2), (2,1), (1,3), (3,1), (2,3), (3,2)
    # That's edges where |i-j| <= 2 and i != j
    assert graph.ii.numel() > 0
    assert graph.jj.numel() > 0
    
    # Verify all edges have |i-j| <= radius
    diff = (graph.ii - graph.jj).abs()
    assert (diff <= 2).all()
    assert (diff > 0).all()
    print("  ✓ Add neighborhood factors")


@torch.inference_mode()
def test_remove_duplicates_empty():
    """Test _remove_duplicates with empty graph."""
    cfg = make_cfg(H=96, W=128)
    graph = FactorGraph(cfg)
    device = torch.device("cuda")
    
    ii = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
    jj = torch.tensor([1, 2, 3], device=device, dtype=torch.long)
    
    # Should return all edges since graph is empty
    ii_out, jj_out = graph._remove_duplicates(ii, jj)
    
    assert ii_out.numel() == 3
    assert jj_out.numel() == 3
    print("  ✓ Remove duplicates (empty graph)")


@torch.inference_mode()
def test_remove_duplicates_with_existing():
    """Test _remove_duplicates with existing edges."""
    cfg = make_cfg(H=96, W=128)
    graph = FactorGraph(cfg)
    device = torch.device("cuda")
    
    T = 5
    payload = make_buffer_payload(T, 96, 128, 8, device, torch.float32)
    
    # Add initial edges
    ii1 = torch.tensor([0, 1], device=device, dtype=torch.long)
    jj1 = torch.tensor([1, 2], device=device, dtype=torch.long)
    graph.add_factors(ii1, jj1, payload)
    
    # Try to add overlapping edges
    ii2 = torch.tensor([0, 2, 3], device=device, dtype=torch.long)
    jj2 = torch.tensor([1, 3, 4], device=device, dtype=torch.long)
    
    ii_out, jj_out = graph._remove_duplicates(ii2, jj2)
    
    # Edge (0,1) should be removed as duplicate
    assert ii_out.numel() == 2
    assert (ii_out == torch.tensor([2, 3], device=device)).all()
    print("  ✓ Remove duplicates (with existing)")


@torch.inference_mode()
def test_remove_factors():
    """Test removing factors from graph."""
    cfg = make_cfg(H=96, W=128)
    graph = FactorGraph(cfg)
    device = torch.device("cuda")
    
    T = 5
    payload = make_buffer_payload(T, 96, 128, 8, device, torch.float32)
    
    # Add 4 edges
    ii = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    jj = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)
    graph.add_factors(ii, jj, payload)
    
    assert graph.ii.numel() == 4
    
    # Remove first 2 edges, store in inactive
    remove_mask = torch.tensor([True, True, False, False], device=device)
    graph.remove_factors(remove_mask, store_inac=True)
    
    assert graph.ii.numel() == 2
    assert graph.ii_inac.numel() == 2
    assert (graph.ii == torch.tensor([2, 3], device=device)).all()
    print("  ✓ Remove factors")


@torch.inference_mode()
def test_remove_keyframe():
    """Test removing a keyframe and adjusting indices."""
    cfg = make_cfg(H=96, W=128)
    graph = FactorGraph(cfg)
    device = torch.device("cuda")
    
    T = 6
    payload = make_buffer_payload(T, 96, 128, 8, device, torch.float32)
    
    # Add edges: (0,1), (1,2), (2,3), (3,4), (4,5)
    ii = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.long)
    jj = torch.tensor([1, 2, 3, 4, 5], device=device, dtype=torch.long)
    graph.add_factors(ii, jj, payload)
    
    # Remove keyframe 2 - edges (1,2) and (2,3) should be removed
    # Remaining edges should have indices decremented for ix >= 2
    graph.remove_keyframe(2)
    
    # After removing kf 2:
    # - Edges involving 2 are removed: (1,2), (2,3) gone
    # - Indices >= 2 decremented: (3,4) -> (2,3), (4,5) -> (3,4)
    # Remaining: (0,1), (2,3), (3,4)
    assert graph.ii.numel() == 3
    print("  ✓ Remove keyframe")


@torch.inference_mode()
def test_corr_block_filtering():
    """Test that CorrBlock pyramid is filtered with edges."""
    cfg = make_cfg(H=96, W=128)
    graph = FactorGraph(cfg)
    device = torch.device("cuda")
    
    T = 5
    payload = make_buffer_payload(T, 96, 128, 8, device, torch.float32)
    
    # Add 4 edges
    ii = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    jj = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)
    graph.add_factors(ii, jj, payload)
    
    # Verify pyramid has 4 edges
    assert graph.corr.pyramid[0].size(0) == 4
    
    # Remove 2 edges
    remove_mask = torch.tensor([True, True, False, False], device=device)
    graph.remove_factors(remove_mask, store_inac=False)
    
    # Pyramid should now have 2 edges
    assert graph.corr.pyramid[0].size(0) == 2
    print("  ✓ CorrBlock filtering")


@torch.inference_mode()
def test_max_factors_limit():
    """Test that max_factors limit is enforced."""
    cfg = make_cfg(H=96, W=128)
    cfg["tracking"]["max_factors"] = 5
    graph = FactorGraph(cfg)
    device = torch.device("cuda")
    
    T = 10
    payload = make_buffer_payload(T, 96, 128, 8, device, torch.float32)
    
    # Add 4 edges
    ii1 = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    jj1 = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)
    graph.add_factors(ii1, jj1, payload)
    
    assert graph.ii.numel() == 4
    
    # Age the existing edges
    graph.age += 5
    
    # Add 3 more edges with remove=True, should trigger removal of oldest
    ii2 = torch.tensor([5, 6, 7], device=device, dtype=torch.long)
    jj2 = torch.tensor([6, 7, 8], device=device, dtype=torch.long)
    graph.add_factors(ii2, jj2, payload, remove=True)
    
    # Should have at most max_factors edges
    assert graph.ii.numel() <= cfg["tracking"]["max_factors"]
    # Removed edges should be in inactive
    assert graph.ii_inac.numel() > 0
    print("  ✓ Max factors limit")


@torch.inference_mode()
def test_frontend_proximity_factors():
    """Test frontend proximity factor addition."""
    cfg = make_cfg(H=96, W=128)
    cfg["tracking"]["max_factors"] = 100
    graph = FactorGraph(cfg)
    device = torch.device("cuda")
    
    T = 8
    payload = make_buffer_payload(T, 96, 128, 8, device, torch.float32)
    
    # Create distance matrix (lower = closer)
    dist = torch.rand(T * T, device=device) * 10
    
    prox_payload = ProximityPayload(
        role=CallerRole.FRONTEND,
        buffer_payload=payload,
        dist=dist,
        t0=0,
        t1=0,
        rad=2,
        nms=2,
        thresh=16.0,
        remove=False,
        t0_loop=None,
    )
    
    graph.add_proximity_factors(prox_payload)
    
    # Should have added some edges
    assert graph.ii.numel() > 0
    print("  ✓ Frontend proximity factors")


@torch.inference_mode()
def test_backend_proximity_factors():
    """Test backend proximity factor addition."""
    cfg = make_cfg(H=96, W=128)
    cfg["tracking"]["max_factors"] = 100
    graph = FactorGraph(cfg)
    device = torch.device("cuda")
    
    T = 8
    payload = make_buffer_payload(T, 96, 128, 8, device, torch.float32)
    
    # When loop=False, t0_loop is reset to t0, so:
    # ilen = t1 - t0_loop = t1 - t0 = T - 0 = T
    # jlen = t1 - t0 = T - 0 = T
    t0, t1 = 0, T
    ilen = t1 - t0
    jlen = t1 - t0
    dist = torch.rand(ilen * jlen, device=device) * 10
    
    prox_payload = ProximityPayload(
        role=CallerRole.BACKEND,
        buffer_payload=payload,
        dist=dist,
        t0=t0,
        t1=t1,
        rad=2,
        nms=2,
        thresh=16.0,
        max_factors=50,
        t0_loop=None,
        loop=False,
        remove=False,
    )
    
    graph.add_proximity_factors(prox_payload)
    
    # Should have added some edges
    assert graph.ii.numel() > 0
    print("  ✓ Backend proximity factors")


@torch.inference_mode()
def test_clear():
    """Test clearing the graph."""
    cfg = make_cfg(H=96, W=128)
    graph = FactorGraph(cfg)
    device = torch.device("cuda")
    
    T = 5
    payload = make_buffer_payload(T, 96, 128, 8, device, torch.float32)
    
    ii = torch.tensor([0, 1], device=device, dtype=torch.long)
    jj = torch.tensor([1, 2], device=device, dtype=torch.long)
    graph.add_factors(ii, jj, payload)
    
    graph.clear()
    
    assert graph.ii is None
    assert graph.jj is None
    assert graph.corr is None
    print("  ✓ Clear")


def run_smoke():
    print("\nFactorGraph Smoke Tests")
    print("=" * 40)
    
    test_initialization()
    test_add_factors()
    test_add_neighborhood_factors()
    test_remove_duplicates_empty()
    test_remove_duplicates_with_existing()
    test_remove_factors()
    test_remove_keyframe()
    test_corr_block_filtering()
    test_max_factors_limit()
    test_frontend_proximity_factors()
    test_backend_proximity_factors()
    test_clear()
    
    print("=" * 40)
    print("All FactorGraph tests passed! ✓\n")


if __name__ == "__main__":
    run_smoke()
