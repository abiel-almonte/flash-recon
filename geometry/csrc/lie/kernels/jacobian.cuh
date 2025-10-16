#pragma once
#include "common/common.h"

Tensor se3_point_jac_cuda(
    Tensor &p
);
__global__ void se3_point_jac_kernel(const float *p, float *j, const int n) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;

    const int idx4 = idx * 4;
    const int idx24 = idx * 24;

    const float x = p[idx4 + 0];
    const float y = p[idx4 + 1];
    const float z = p[idx4 + 2];
    const float d = p[idx4 + 3];

    // Row 1
    j[idx24] = d;
    j[idx24 + 4] = z;
    j[idx24 + 5] = -y;

    // Row 2
    j[idx24 + 7] = d;
    j[idx24 + 9] = -z;
    j[idx24 + 11] = x;

    // Row 3
    j[idx24 + 14] = d;
    j[idx24 + 15] = y;
    j[idx24 + 16] = -x;
}

// Apply Adjoint^T to jacobians: jac [B, S, 2, 6], returns same shape
// Each row split [rho(3) | phi(3)]
// rho' = rho * R^T
// phi' = (phi - t x rho) * R^T
Tensor se3_adjointT_cuda(
    Tensor &t,
    Tensor &q,
    Tensor &jac
);
__global__ void se3_adjointT_kernel(const float* t, const float* q, const float* jac, float* out, const int B, const int S) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // over B*S*2 rows
    const int rows_per_b = S * 2;
    const int total_rows = B * rows_per_b;

    if (idx >= total_rows) {
        return;
    }

    const int b = idx / rows_per_b;
    const int rs = idx % rows_per_b; // row within [S*2]
    const int s = rs / 2;
    const int r = rs % 2; // 0 or 1

    // Load pose
    const float tx = t[b * 3 + 0];
    const float ty = t[b * 3 + 1];
    const float tz = t[b * 3 + 2];
    const float qx = q[b * 4 + 0];
    const float qy = q[b * 4 + 1];
    const float qz = q[b * 4 + 2];
    const float qw = q[b * 4 + 3];

    // Build R^T from q on the fly
    const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const float xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const float wx = qw * qx, wy = qw * qy, wz = qw * qz;

    // R = [[1-2(yy+zz), 2(xy-wz), 2(xz+wy)],
    //      [2(xy+wz), 1-2(xx+zz), 2(yz-wx)],
    //      [2(xz-wy), 2(yz+wx), 1-2(xx+yy)]]
    // R^T = transpose(R)
    const float RT[9] = {
        1.f - 2.f * (yy + zz),
        2.f * (xy + wz),
        2.f * (xz - wy),
        2.f * (xy - wz),
        1.f - 2.f * (xx + zz),
        2.f * (yz + wx),
        2.f * (xz + wy),
        2.f * (yz - wx),
        1.f - 2.f * (xx + yy)
    };

    const int base = ((b * S + s) * 2 + r) * 6;

    // Input
    const float rho0 = jac[base + 0];
    const float rho1 = jac[base + 1];
    const float rho2 = jac[base + 2];
    const float phi0 = jac[base + 3];
    const float phi1 = jac[base + 4];
    const float phi2 = jac[base + 5];

    // rho' = rho * R^T
    float rhoR[3];
    for (int j = 0; j < 3; ++j) {
        const int j3 = j * 3;
        rhoR[j] = rho0 * RT[j3 + 0] + rho1 * RT[j3 + 1] + rho2 * RT[j3 + 2];
    }

    // phi' = (phi - t x rho) * R^T
    float px = phi0 - (ty * rho2 - tz * rho1);
    float py = phi1 - (tz * rho0 - tx * rho2);
    float pz = phi2 - (tx * rho1 - ty * rho0);

    float phiR[3];
    for (int j = 0; j < 3; ++j) {
        const int j3 = j * 3;
        phiR[j] = px * RT[j3 + 0] + py * RT[j3 + 1] + pz * RT[j3 + 2];
    }

    out[base + 0] = rhoR[0];
    out[base + 1] = rhoR[1];
    out[base + 2] = rhoR[2];
    out[base + 3] = phiR[0];
    out[base + 4] = phiR[1];
    out[base + 5] = phiR[2];
}
