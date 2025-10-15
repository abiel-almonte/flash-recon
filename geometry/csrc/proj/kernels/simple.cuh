#pragma once
#include "common/common.h"

template<typename T, int N>
using PackedAccessor = typename torch::PackedTensorAccessor32<T, N, at::RestrictPtrTraits>;

Tensor proj_cuda(
    Tensor p,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const int last_dim
);
__global__ void proj_kernel(
    const float *p,
    float *c,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const int chunk_size,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const int idx4 = idx * 4;
    const int idx_out = idx * chunk_size;

    const float x = p[idx4 + 0];
    const float y = p[idx4 + 1];
    const float z = (p[idx4 + 2] < HALF_MINDEPTH) ? 1.0f : p[idx4 + 2];
    const float d = p[idx4 + 3];

    const float a = 1.0f / z;

    const float x_adjusted = fx * (x * a) + cx;
    const float y_adjusted = fy * (y * a) + cy;

    c[idx_out + 0] = x_adjusted;
    c[idx_out + 1] = y_adjusted;

    if (chunk_size > 2) {
        c[idx_out + 2] = d * a;
    }
}

Tensor proj_jac_cuda(
    Tensor p,
    const float fx,
    const float fy,
    const float cx,
    const float cy
);
__global__ void proj_jac_kernel(
    const float *p,
    float *j,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const int idx4 = idx * 4;
    const int idx8 = idx * 8;

    const float x = p[idx4 + 0];
    const float y = p[idx4 + 1];
    const float z = (p[idx4 + 2] < HALF_MINDEPTH) ? 1.0f : p[idx4 + 2];
    const float d = p[idx4 + 3];

    const float a = 1.0f / z;
    const float a2 = a * a;

    j[idx8 + 0] = fx * a;
    j[idx8 + 2] = -fx * x * a2;

    j[idx8 + 5] = fy * a;
    j[idx8 + 6] = -fy * y * a2;
}
