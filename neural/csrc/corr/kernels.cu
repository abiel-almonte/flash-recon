#include "kernels.h"
#include "kernels/corr.cuh"
#include "kernels/altcorr.cuh"


Tensor corr_forward(
    const Tensor volume, // [T, H, W, Hi, Wi]
    const Tensor coords, // [T, 2, H, W]
    const int radius
){
    CHECK_DEVICE(volume);
    CHECK_CONTIGUOUS(volume);
    CHECK_INPUT(coords);

    const int T = volume.size(0);
    const int H = volume.size(1);
    const int W = volume.size(2);
    const int rd = 2*radius + 1;

    auto opts = coords.options();
    Tensor corr = torch::zeros({T, rd, rd, H, W}, opts);
    
    const dim3 blocks((W + 16 - 1) / 16, (H + 16 - 1) / 16, T);
    const dim3 threads(16, 16);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.scalar_type(), "corr_forward", [&] {
        corr_forward_kernel<scalar_t><<<blocks, threads>>>(
            volume.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
            coords.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            corr.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            radius
        );
    });

    corr = corr.view({T, rd*rd,H, W});
    return corr;
}


Tensor altcorr_forward(
    const Tensor fmap1,
    const Tensor fmap2,
    const Tensor coords,
    int radius
) { 
    CHECK_INPUT(fmap1);
    CHECK_INPUT(fmap2);
    CHECK_INPUT(coords);

    const int T = coords.size(0);
    const int H = coords.size(2);
    const int W = coords.size(3);

    const int rd = 2 * radius + 1;
    
    auto opts = fmap1.options();
    Tensor corr = torch::zeros({T, rd * rd, H, W}, opts);

    const dim3 blocks(T, (H + 4 - 1) / 4, (W + 8 - 1) / 8);
    const dim3 threads(4, 8);
      
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), "altcorr_forward_kernel", [&] {
        altcorr_forward_kernel<scalar_t><<<blocks, threads>>>(
            fmap1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            fmap2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            corr.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            radius
        );
      }
    );

    return corr;
}
