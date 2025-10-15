#pragma once
#include "common/common.h"

template <typename T, int N>
using PackedAccessor = typename torch::PackedTensorAccessor32<T, N, at::RestrictPtrTraits>;

__forceinline__ __device__ bool within_bounds(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

Tensor corr_forward(
    const Tensor volume, // [T, H, W, Hi, Wi]
    const Tensor coords, // [T, H, W, 2]
    const int radius
);
__global__ void corr_forward_kernel(
    const PackedAccessor<float, 5> volume,
    const PackedAccessor<float, 4> coords,
    PackedAccessor<float, 5> corr_out,
    const int r)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;

    const int h1 = volume.size(1);
    const int w1 = volume.size(2);
    const int h2 = volume.size(3);
    const int w2 = volume.size(4);

    if (!within_bounds(y, x, h1, w1)) {
        return;
    }

    float x0 = coords[n][0][y][x];
    float y0 = coords[n][1][y][x];

    float dx = x0 - floor(x0);
    float dy = y0 - floor(y0);

    int rd = 2 * r + 1;
    for (int i = 0; i < rd + 1; i++) {
        for (int j = 0; j < rd + 1; j++) {
            int x1 = static_cast<int>(floor(x0)) - r + i;
            int y1 = static_cast<int>(floor(y0)) - r + j;

            if (within_bounds(y1, x1, h2, w2)) {
                float s = volume[n][y][x][y1][x1];

                if (i > 0 && j > 0)
                    corr_out[n][i - 1][j - 1][y][x] += s * float(dx * dy);

                if (i > 0 && j < rd)
                    corr_out[n][i - 1][j][y][x] += s * float(dx * (1.0f - dy));

                if (i < rd && j > 0)
                    corr_out[n][i][j - 1][y][x] += s * float((1.0f - dx) * dy);

                if (i < rd && j < rd)
                    corr_out[n][i][j][y][x] += s * float((1.0f - dx) * (1.0f - dy));
            }
        }
    }
}
