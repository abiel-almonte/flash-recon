#pragma once
#include "common/common.h"
#include "common/device_ops.cuh"

template<typename T, int N>
using PackedAccessor = typename torch::PackedTensorAccessor32<T, N, at::RestrictPtrTraits>;

Tensor frame_distance_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj,
    float beta
);

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULLMASK, val, offset);
    return val;
}

__device__ inline float blockReduceSum(float val) {
    __shared__ float shared[32]; // one per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Only first warp participates in final reduction
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

__global__ void frame_distance_kernel(
    const PackedAccessor<float, 2> t,
    const PackedAccessor<float, 2> q,
    const PackedAccessor<float, 3> disps,
    const PackedAccessor<float, 1> intr,
    const PackedAccessor<long, 1> ii,
    const PackedAccessor<long, 1> jj,
    PackedAccessor<float, 1> dist_out,
    const float beta
) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x;

    const int ht = disps.size(1);
    const int wd = disps.size(2);

    __shared__ int ix, jx;
    __shared__ float fx, fy, cx, cy;
    __shared__ float ti[3], tj[3], tij[3];
    __shared__ float qi_s[4], qj_s[4], qij[4];

    if (tid == 0) {
        setup_projective_transform(
            e,
            t,
            q,
            intr,
            ii,
            jj,
            &ix,
            &jx,
            &fx,
            &fy,
            &cx,
            &cy,
            ti,
            qi_s,
            tj,
            qj_s,
            tij,
            qij
        );
        
    }
    __syncthreads();

    float local_accum = 0.0f;
    float local_valid = 0.0f;

    for (int k = tid; k < ht * wd; k += blockDim.x) {
        const int row = k / wd;
        const int col = k % wd;
        const float u = static_cast<float>(col);
        const float v = static_cast<float>(row);
        const float d = disps[ix][row][col];

        float Xi[4];
        unproject_pixel(u, v, d, fx, fy, cx, cy, Xi);

        // Skip invalid depth
        if (Xi[2] * d < 0.001f) continue;

        // Full SE3 path (weight = beta)
        float Xj[4];
        actSE3(tij, qij, Xi, Xj);

        if (Xj[2] > 0.01f) {
            float uj, vj;
            project_point(Xj[0], Xj[1], Xj[2], fx, fy, cx, cy, &uj, &vj, 0.01f);
            float du = uj - u;
            float dv = vj - v;
            float flow_se3 = sqrtf(du * du + dv * dv);

            // Translation-only path (weight = 1-beta)
            // Apply only translation to Xi: Xt = Xi + h*tij (no rotation)
            float Xt[4];
            Xt[0] = Xi[0] + Xi[3] * tij[0];
            Xt[1] = Xi[1] + Xi[3] * tij[1];
            Xt[2] = Xi[2] + Xi[3] * tij[2];
            Xt[3] = Xi[3];

            float ut, vt;
            if (Xt[2] > 0.01f) {
                project_point(Xt[0], Xt[1], Xt[2], fx, fy, cx, cy, &ut, &vt, 0.01f);
                float dut = ut - u;
                float dvt = vt - v;
                float flow_t = sqrtf(dut * dut + dvt * dvt);

                local_accum += beta * flow_se3 + (1.0f - beta) * flow_t;
                local_valid += 1.0f;
            }
        }
    }

    // Block reduction
    float total_accum = blockReduceSum(local_accum);
    float total_valid = blockReduceSum(local_valid);

    if (tid == 0) {
        float total_pixels = static_cast<float>(ht * wd);
        if (total_valid / total_pixels > 0.75f) {
            dist_out[e] = total_accum / total_valid;
        } else {
            dist_out[e] = 1000.0f;
        }
    }
}
