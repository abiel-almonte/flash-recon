#include "kernels.h"
#include "kernels/simple.cuh"
#include "kernels/fused.cuh"
#include "kernels/distance.cuh"

Tensor proj_cuda(Tensor &p, const float fx, const float fy, const float cx, const float cy, const int last_dim) {
    CHECK_INPUT(p);

    const int n = p.size(0);
    auto coords = torch::zeros({n, last_dim}, p.options());

    const int blocks = (n + THREADS - 1) / THREADS;

    proj_kernel<<<blocks, THREADS>>>(
        p.data_ptr<float>(),
        coords.data_ptr<float>(),
        fx,
        fy,
        cx,
        cy,
        last_dim,
        n
    );

    return coords;
}

Tensor proj_jac_cuda(Tensor &p, const float fx, const float fy, const float cx, const float cy) {
    CHECK_INPUT(p);

    const int n = p.size(0);
    auto jacobian = torch::zeros({n, 2, 4}, p.options());

    const int blocks = (n + THREADS - 1) / THREADS;

    proj_jac_kernel<<<blocks, THREADS>>>(
        p.data_ptr<float>(),
        jacobian.data_ptr<float>(),
        fx,
        fy,
        cx,
        cy,
        n
    );

    return jacobian;
}

std::vector<Tensor> fused_projective_jac_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj
) {
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrinsics);
    CHECK_INPUTL(ii);
    CHECK_INPUTL(jj);

    const int E = ii.size(0);
    const int H = disps.size(1);
    const int W = disps.size(2);

    auto opts = disps.options();
    auto half_opts = opts.dtype(torch::kFloat16);

    Tensor coords = torch::zeros({E, H, W, 2}, opts);
    Tensor valid = torch::zeros({E, H, W, 1}, opts);
    Tensor Xj_cache = torch::zeros({E, H, W, 4}, opts);
    
    Tensor Ji = torch::zeros({E, H, W, 2, 6}, opts);
    Tensor Jj = torch::zeros({E, H, W, 2, 6}, opts);
    Tensor Jz = torch::zeros({E, H, W, 2, 1}, opts);

    fused_projective_with_cache_kernel<<<E, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        valid.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        Xj_cache.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );

    projective_jacobians_kernel<<<E, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        valid.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        Xj_cache.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        Ji.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        Jj.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        Jz.packed_accessor32<float, 5, torch::RestrictPtrTraits>()
    );

    return {coords, valid, Ji, Jj, Jz};
}

std::vector<Tensor> fused_projective_cuda(
    Tensor t, // [B,3]
    Tensor q, // [B,4]
    Tensor disps, // [B,H,W]
    Tensor intrinsics, // [4]
    Tensor ii, Tensor jj
) {
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrinsics);
    CHECK_INPUTL(ii);
    CHECK_INPUTL(jj);

    const int E = ii.size(0);
    const int H = disps.size(1);
    const int W = disps.size(2);

    auto opts = disps.options();
    Tensor coords = torch::zeros({E, H, W, 2}, opts);
    Tensor valid = torch::zeros({E, H, W, 1}, opts);

    fused_projective_kernel<false><<<E, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        valid.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );

    return {coords, valid};
}


std::vector<Tensor> fused_induced_flow_cuda(
    Tensor t, // [B,3]
    Tensor q, // [B,4]
    Tensor disps, // [B,H,W]
    Tensor intrinsics, // [4]
    Tensor ii,
    Tensor jj
) {
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrinsics);
    CHECK_INPUTL(ii);
    CHECK_INPUTL(jj);

    const int E = ii.size(0);
    const int H = disps.size(1);
    const int W = disps.size(2);

    auto opts = disps.options();
    Tensor coords = torch::zeros({E, H, W, 2}, opts);
    Tensor valid = torch::zeros({E, H, W, 1}, opts);

    fused_projective_kernel<true><<<E, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        valid.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );

    return {coords, valid};
}

Tensor fused_depth_filter_cuda(
    Tensor t, // [T, 3]
    Tensor q, // [T, 4]
    Tensor disps, // [T, H, W]
    Tensor intrinsics, // [4]
    Tensor ii, // [M]
    Tensor thresh // [M]
) { 
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrinsics);
    CHECK_INPUTL(ii);
    CHECK_INPUT(thresh);

    const int M = ii.size(0);
    const int H = disps.size(1);
    const int W = disps.size(2);

    auto opts = disps.options();
    Tensor count = torch::zeros({M, H, W}, opts);

    dim3 grid(M, 6, (H*W + THREADS - 1) / THREADS); // 6 views max
    depth_filter_kernel<<<grid, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        thresh.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        count.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
    );

    return count;
}

Tensor frame_distance_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj,
    float beta
) {
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrinsics);
    CHECK_INPUTL(ii);
    CHECK_INPUTL(jj);

    const int E = ii.size(0);
    auto opts = disps.options();
    Tensor dist = torch::zeros({E}, opts);

    frame_distance_kernel<<<E, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        dist.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        beta
    );

    return dist;
}
