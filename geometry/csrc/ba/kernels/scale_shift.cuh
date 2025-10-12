#pragma once
#include "common/common.h"
#include "common/device_ops.cuh"

template<typename T, int N>
using PackedAccessor = typename torch::PackedTensorAccessor32<T, N, at::RestrictPtrTraits>;

std::tuple<Tensor, Tensor> fused_projective_transform_with_reduction_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj,
    Tensor target,
    Tensor weight
);
__global__ void fused_projective_transform_with_reduction_kernel(
    PackedAccessor<float, 2> t,
    PackedAccessor<float, 2> q,
    PackedAccessor<float, 3> disps,
    PackedAccessor<float, 1> intr,
    PackedAccessor<long, 1> ii,
    PackedAccessor<long, 1> jj,
    PackedAccessor<float, 4> target,
    PackedAccessor<float, 4> weight,
    PackedAccessor<float, 2> curv,
    PackedAccessor<float, 2> rhs
) {
    const int e = blockIdx.x; // edge index
    const int tid = threadIdx.x;

    const int ht = disps.size(1);
    const int wd = disps.size(2);

    __shared__ int ix;
    __shared__ int jx;
    __shared__ float fx, fy, cx, cy;
    __shared__ float ti[3], tj[3], tij[3];
    __shared__ float qi[4], qj[4], qij[4];

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
            qi,
            tj,
            qj,
            tij,
            qij
        );
    }

    __syncthreads();

    for (int k = tid; k < ht * wd; k += blockDim.x) {
        const int i = k / wd;
        const int j = k % wd;
        const float u = static_cast<float>(j);
        const float v = static_cast<float>(i);

        float Xi[4];
        unproject_pixel(
            u,
            v,
            disps[ix][i][j],
            fx,
            fy,
            cx,
            cy,
            Xi
        );

        float Xj[4];
        actSE3(tij, qij,
               Xi, Xj);

        float coords_temp[2];
        float r_temp[2];

        coords_temp[0] = u;
        coords_temp[1] = v;
        project_point(
            Xj[0],
            Xj[1],
            Xj[2],
            fx,
            fy,
            cx,
            cy,
            &coords_temp[0],
            &coords_temp[1],
            0.01f
        );

        r_temp[0] = target[e][i][j][0] - coords_temp[0];
        r_temp[1] = target[e][i][j][1] - coords_temp[1];

        if (Xj[2] > 0.2f && Xi[2] > 0.2f){ //is valid
            float Jz_temp[2];
            compute_depth_jacobian(
                Xj[0],
                Xj[1],
                Xj[2],
                tij,
                fx,
                fy,
                Jz_temp
            );

            for(int ii = 0; ii < 2; ii++){
                const float w = 0.001f * weight[e][i][j][ii];
                const int iwd_j = i*wd + j;
                curv[e][iwd_j] += w * (-Jz_temp[ii]) * (-Jz_temp[ii]);
                rhs[e][iwd_j] -= w * r_temp[ii] * Jz_temp[ii];
            }

        }
    }
}

std::tuple<Tensor, Tensor, Tensor> fused_depth_jacobians_cuda(
    Tensor disps,
    Tensor mono_disps,
    Tensor valid_depth,
    Tensor scales,
    Tensor shifts,
    Tensor ignore,
    const float alpha
);
__global__ void fused_depth_jacobians_kernel(
    PackedAccessor<float, 3> disps,
    PackedAccessor<float, 3> mono_disps,
    PackedAccessor<bool, 3> valid_depth,
    PackedAccessor<float, 1> scales,
    PackedAccessor<float, 1> shifts,
    PackedAccessor<bool, 1> ignore,
    const float sqrt_alpha,
    const float sqrt_alpha10,
    PackedAccessor<float, 3> Jwq_out,
    PackedAccessor<float, 2> Jd_out,
    PackedAccessor<float, 2> Rd_out
) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x;

    const int ht = disps.size(1);
    const int wd = disps.size(2);

    __shared__ bool is_ignored;
    __shared__ float scale;
    __shared__ float shift;

    if (tid == 0) {
        is_ignored = ignore[e];
        scale = scales[e];
        shift = shifts[e];
    }
    __syncthreads();

    for(int k = tid; k < ht*wd; k += blockDim.x) {
        const int i = k /wd;
        const int j = k % wd;
        const int iwd_j = i*wd + j;

        const bool is_valid_depth = valid_depth[e][i][j];
        
        const float Jd = (is_valid_depth)? sqrt_alpha10 : sqrt_alpha;
        const float mono = mono_disps[e][i][j];
        const float disp = disps[e][i][j];
        
        const bool is_invalid = is_ignored || (mono < 1e-6f);

        Rd_out[e][iwd_j] = sqrt_alpha * (disp - (scale*mono + shift)); // depth residual

        if (is_invalid){
            if (!is_valid_depth) {
                Jd_out[e][iwd_j] = Jd;
            }
        } else{
            Jd_out[e][iwd_j] = Jd;
            Jwq_out[e][iwd_j][0] = -mono * Jd; // scale
            Jwq_out[e][iwd_j][1] = -Jd; // shift
        }
    }
}
