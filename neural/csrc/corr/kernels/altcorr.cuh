// Adapted from DROID-SLAM (BSD 3-Clause)
#pragma once
#include "common/common.h"

#define ALT_BLOCK_H 4
#define ALT_BLOCK_W 8
#define ALT_BLOCK_HW (ALT_BLOCK_H * ALT_BLOCK_W)
#define ALT_CHANNEL_STRIDE 32

__forceinline__ __device__
bool alt_within_bounds(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void altcorr_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> corr,
    int r)
{
    const int b  = blockIdx.x;
    const int h0 = blockIdx.y * blockDim.x;
    const int w0 = blockIdx.z * blockDim.y;
    const int tid = threadIdx.x * blockDim.y + threadIdx.y;

    const int H1 = fmap1.size(1);
    const int W1 = fmap1.size(2);
    const int H2 = fmap2.size(1);
    const int W2 = fmap2.size(2);
    const int C  = fmap1.size(3);

    __shared__ scalar_t f1[ALT_CHANNEL_STRIDE][ALT_BLOCK_HW];
    __shared__ scalar_t f2[ALT_CHANNEL_STRIDE][ALT_BLOCK_HW];
    __shared__ float x2s[ALT_BLOCK_HW];
    __shared__ float y2s[ALT_BLOCK_HW];

    for (int c = 0; c < C; c += ALT_CHANNEL_STRIDE) {
        // load fmap1 tile into shared memory
        for (int k = 0; k < ALT_BLOCK_HW; k += ALT_BLOCK_HW / ALT_CHANNEL_STRIDE) {
            int k1 = k + tid / ALT_CHANNEL_STRIDE;
            int h1 = h0 + k1 / ALT_BLOCK_W;
            int w1 = w0 + k1 % ALT_BLOCK_W;
            int c1 = tid % ALT_CHANNEL_STRIDE;

            f1[c1][k1] = alt_within_bounds(h1, w1, H1, W1)
                ? fmap1[b][h1][w1][c + c1]
                : static_cast<scalar_t>(0);
        }

        __syncthreads();

        // coords layout: [T, 2, H, W] — channel 0 = x, channel 1 = y
        int h1 = h0 + threadIdx.x;
        int w1 = w0 + threadIdx.y;
        if (alt_within_bounds(h1, w1, H1, W1)) {
            x2s[tid] = coords[b][0][h1][w1];
            y2s[tid] = coords[b][1][h1][w1];
        }

        float dx = x2s[tid] - floorf(x2s[tid]);
        float dy = y2s[tid] - floorf(y2s[tid]);

        int rd = 2 * r + 1;
        for (int iy = 0; iy < rd + 1; iy++) {
            for (int ix = 0; ix < rd + 1; ix++) {
                // load fmap2 tile for this offset
                for (int k = 0; k < ALT_BLOCK_HW; k += ALT_BLOCK_HW / ALT_CHANNEL_STRIDE) {
                    int k1 = k + tid / ALT_CHANNEL_STRIDE;
                    int h2 = static_cast<int>(floorf(y2s[k1])) - r + iy;
                    int w2 = static_cast<int>(floorf(x2s[k1])) - r + ix;
                    int c2 = tid % ALT_CHANNEL_STRIDE;

                    f2[c2][k1] = alt_within_bounds(h2, w2, H2, W2)
                        ? fmap2[b][h2][w2][c + c2]
                        : static_cast<scalar_t>(0);
                }

                __syncthreads();

                // dot product over channel chunk
                float s = 0.0f;
                for (int k = 0; k < ALT_CHANNEL_STRIDE; k++)
                    s += static_cast<float>(f1[k][tid]) * static_cast<float>(f2[k][tid]);

                // bilinear splat into output
                int ix_nw = H1 * W1 * ((iy - 1) + rd * (ix - 1));
                int ix_ne = H1 * W1 * ((iy - 1) + rd * ix);
                int ix_sw = H1 * W1 * (iy + rd * (ix - 1));
                int ix_se = H1 * W1 * (iy + rd * ix);

                float nw = s * (dy) * (dx);
                float ne = s * (dy) * (1.0f - dx);
                float sw = s * (1.0f - dy) * (dx);
                float se = s * (1.0f - dy) * (1.0f - dx);

                float* corr_ptr = &corr[b][0][h1][w1];

                if (iy > 0 && ix > 0 && alt_within_bounds(h1, w1, H1, W1))
                    *(corr_ptr + ix_nw) += nw;

                if (iy > 0 && ix < rd && alt_within_bounds(h1, w1, H1, W1))
                    *(corr_ptr + ix_ne) += ne;

                if (iy < rd && ix > 0 && alt_within_bounds(h1, w1, H1, W1))
                    *(corr_ptr + ix_sw) += sw;

                if (iy < rd && ix < rd && alt_within_bounds(h1, w1, H1, W1))
                    *(corr_ptr + ix_se) += se;
            }
        }
    }
}
