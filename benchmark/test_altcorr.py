import torch

from neural import Corr, AltCorr


def test_kernel_vs_bmm():
    """Test altcorr_forward kernel against bmm + corr_forward for a single level."""
    torch.manual_seed(42)
    T, C, ht, wd = 4, 128, 48, 64
    radius = 3

    fmap1 = torch.randn(T, C, ht, wd, device="cuda")
    fmap2 = torch.randn(T, C, ht, wd, device="cuda")

    # Random coords in valid range
    coords = torch.empty(T, ht, wd, 2, device="cuda")
    coords[..., 0] = torch.rand(T, ht, wd, device="cuda") * (ht - 1)
    coords[..., 1] = torch.rand(T, ht, wd, device="cuda") * (wd - 1)

    # --- CorrBlock path ---
    corr = Corr({"corrblock": {"num_levels": 1, "radius": radius}})
    corr.build_pyramid(fmap1, fmap2)
    out_corr = corr(coords)

    # --- AltCorrBlock ---
    fmaps = torch.cat([fmap1, fmap2], dim=0)
    ii = torch.arange(T, device="cuda", dtype=torch.long)
    jj = torch.arange(T, 2 * T, device="cuda", dtype=torch.long)

    altcorr = AltCorr({})
    altcorr.build_pyramid(fmaps)
    out_alt = altcorr(coords, ii, jj)

    diff = (out_corr - out_alt).abs()
    rel_diff = diff / (out_corr.abs() + 1e-8)

    print("=== Kernel vs BMM (single level) ===")
    print(
        f"  CorrBlock  range: [{out_corr.min():.4f}, {out_corr.max():.4f}], mean={out_corr.mean():.6f}"
    )
    print(
        f"  AltCorr    range: [{out_alt.min():.4f}, {out_alt.max():.4f}], mean={out_alt.mean():.6f}"
    )
    print(f"  Abs diff:  max={diff.max():.6f}, mean={diff.mean():.6f}")
    print(f"  Rel diff:  max={rel_diff.max():.6f}, mean={rel_diff.mean():.6f}")
    print(f"  Match (atol=0.1): {torch.allclose(out_corr, out_alt, atol=0.1)}")
    print(f"  Match (atol=0.01): {torch.allclose(out_corr, out_alt, atol=0.01)}")
    print()


def test_full_pipeline():
    """Test full AltCorrBlock (multi-level) vs CorrBlock."""
    torch.manual_seed(42)
    T, C, ht, wd = 4, 128, 48, 64
    radius = 3
    num_levels = 4
    cfg = {"corrblock": {"num_levels": num_levels, "radius": radius}}

    fmap1 = torch.randn(T, C, ht, wd, device="cuda")
    fmap2 = torch.randn(T, C, ht, wd, device="cuda")

    coords = torch.empty(T, ht, wd, 2, device="cuda")
    coords[..., 0] = torch.rand(T, ht, wd, device="cuda") * (ht - 1)
    coords[..., 1] = torch.rand(T, ht, wd, device="cuda") * (wd - 1)

    # --- Corr ---
    corr_block = Corr(cfg)
    corr_block.build_pyramid(fmap1, fmap2)
    out_corr = corr_block(coords)

    # --- AltCorr ---
    # Simulate buffer fmaps: [N, 1, C, ht, wd]
    N = 8  # pretend we have 8 keyframes
    fmaps = torch.zeros(N, C, ht, wd, device="cuda")
    ii = torch.arange(T, device="cuda", dtype=torch.long)
    jj = torch.arange(T, device="cuda", dtype=torch.long) + T
    fmaps[ii] = fmap1
    fmaps[jj] = fmap2

    altcorr = AltCorr({})
    altcorr.build_pyramid(fmaps)
    out_alt = altcorr(coords, ii, jj)

    diff = (out_corr - out_alt).abs()
    K = (2 * radius + 1) ** 2
    print("=== Full pipeline (multi-level) ===")
    print(f"  CorrBlock  range: [{out_corr.min():.4f}, {out_corr.max():.4f}]")
    print(f"  AltCorr    range: [{out_alt.min():.4f}, {out_alt.max():.4f}]")
    print(f"  Abs diff:  max={diff.max():.6f}, mean={diff.mean():.6f}")
    for lvl in range(num_levels):
        d = diff[:, lvl * K : (lvl + 1) * K]
        c = out_corr[:, lvl * K : (lvl + 1) * K]
        a = out_alt[:, lvl * K : (lvl + 1) * K]
        print(
            f"  Level {lvl}: corr=[{c.min():.4f},{c.max():.4f}] alt=[{a.min():.4f},{a.max():.4f}] maxdiff={d.max():.6f}"
        )
    print(f"  Match (atol=1e-3): {torch.allclose(out_corr, out_alt, atol=1e-3)}")
    print()


def test_identity_coords():
    """Test with identity coords (each pixel maps to itself) — correlation should be self-dot-product."""
    torch.manual_seed(42)
    T, C, ht, wd = 2, 128, 48, 64
    radius = 3
    cfg = {"corrblock": {"num_levels": 1, "radius": radius}}

    fmap = torch.randn(T, C, ht, wd, device="cuda")

    # Identity coords: each pixel maps to itself
    y, x = torch.meshgrid(
        torch.arange(ht, device="cuda", dtype=torch.float),
        torch.arange(wd, device="cuda", dtype=torch.float),
        indexing="ij",
    )
    coords = torch.stack([x, y], dim=-1).unsqueeze(0).expand(T, -1, -1, -1).contiguous()

    # Corr
    corr_block = Corr(cfg)
    corr_block.build_pyramid(fmap, fmap)
    out_corr = corr_block(coords)

    # AltCorr
    ii = torch.arange(T, device="cuda", dtype=torch.long)
    jj = torch.arange(T, device="cuda", dtype=torch.long)
    altcorr = AltCorr({})
    altcorr.build_pyramid(fmap)
    out_alt = altcorr(coords, ii, jj)

    # Center bin (radius, radius) should be the self-dot-product / 16
    rd = 2 * radius + 1
    center_idx = radius * rd + radius  # flat index for center bin

    expected_center = (fmap**2).sum(dim=1) / 16.0  # [T, ht, wd]
    corr_center = out_corr[:, center_idx]
    alt_center = out_alt[:, center_idx]

    print("=== Identity coords (self-correlation) ===")
    print(
        f"  Expected center: [{expected_center.min():.4f}, {expected_center.max():.4f}], mean={expected_center.mean():.4f}"
    )
    print(
        f"  CorrBlock center: [{corr_center.min():.4f}, {corr_center.max():.4f}], mean={corr_center.mean():.4f}"
    )
    print(
        f"  AltCorr center: [{alt_center.min():.4f}, {alt_center.max():.4f}], mean={alt_center.mean():.4f}"
    )
    print(
        f"  Diff(expected, corr): max={( expected_center - corr_center).abs().max():.6f}"
    )
    print(
        f"  Diff(expected, alt):  max={(expected_center - alt_center).abs().max():.6f}"
    )
    print(f"  Diff(corr, alt):      max={(corr_center - alt_center).abs().max():.6f}")
    print()


if __name__ == "__main__":
    test_identity_coords()
    test_kernel_vs_bmm()
    test_full_pipeline()
