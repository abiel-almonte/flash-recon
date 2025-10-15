#pragma once
#include "common/common.h"

Tensor se3_transform3d_cuda(
    Tensor &t,
    Tensor &q,
    Tensor &points
);
Tensor se3_transform4d_cuda(
    Tensor &t,
    Tensor &q,
    Tensor &points
);
template <bool HOMOG>
__global__ void se3_transform_kernel(const float *t, const float *q, const float *pts, float *out, const int B, const int S) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = B * S;

    if (idx >= N) {
        return;
    }

    const int b = idx / S;
    const int s = idx % S;
    const int D = HOMOG ? 4 : 3;
    const int base = (b * S + s) * D;

    const float vx = pts[base + 0];
    const float vy = pts[base + 1];
    const float vz = pts[base + 2];
    const float w = HOMOG ? pts[base + 3] : 1.0f;

    const float tx = t[b * 3 + 0];
    const float ty = t[b * 3 + 1];
    const float tz = t[b * 3 + 2];

    const float qx = q[b * 4 + 0];
    const float qy = q[b * 4 + 1];
    const float qz = q[b * 4 + 2];
    const float qw = q[b * 4 + 3];

    float rx, ry, rz;
    const float cx1 = qy * vz - qz * vy;
    const float cy1 = qz * vx - qx * vz;
    const float cz1 = qx * vy - qy * vx;

    const float cx2 = qy * cz1 - qz * cy1;
    const float cy2 = qz * cx1 - qx * cz1;
    const float cz2 = qx * cy1 - qy * cx1;

    rx = vx + 2.0f * (qw * cx1 + cx2);
    ry = vy + 2.0f * (qw * cy1 + cy2);
    rz = vz + 2.0f * (qw * cz1 + cz2);

    out[base + 0] = rx + tx * w;
    out[base + 1] = ry + ty * w;
    out[base + 2] = rz + tz * w;

    if (HOMOG){
        out[base + 3] = w;
    }
}
