#include "kernels.h"
#include "kernels/corr.cuh"

Tensor corr_forward(
    const Tensor volume, // [T, H, W, Hi, Wi]
    const Tensor coords, // [T, 2, H, W]
    const int radius
){
    CHECK_INPUT(volume);
    CHECK_INPUT(coords);

    const int T = volume.size(0);
    const int H = volume.size(1);
    const int W = volume.size(2);
    const int rd = 2*radius + 1;

    auto opts = coords.options();
    Tensor corr = torch::zeros({T, rd, rd, H, W}, opts);
    
    const dim3 blocks((W + 16 - 1) / 16, (H + 16 - 1) / 16, T);
    const dim3 threads(16, 16);
    
    corr_forward_kernel<<<blocks, threads>>>(
        volume.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        corr.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        radius
    );

    corr = corr.view({T, rd*rd,H, W});
    return corr;
}

