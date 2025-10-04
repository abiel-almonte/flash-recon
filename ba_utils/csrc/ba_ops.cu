#include "common.h"

__device__ inline void quat_rotate_vec(const float *q, const float *v, float *r) {
    const float qx = q[0];
    const float qy = q[1];
    const float qz = q[2]; 
    const float qw = q[3];

    const float vx = v[0];
    const float vy = v[1];
    const float vz = v[2];

    const float cx1x = qy * vz - qz * vy;
    const float cx1y = qz * vx - qx * vz;
    const float cx1z = qx * vy - qy * vx;
    const float cx2x = qy * cx1z - qz * cx1y;
    const float cx2y = qz * cx1x - qx * cx1z;
    const float cx2z = qx * cx1y - qy * cx1x;

    r[0] = vx + 2.0f * (qw * cx1x + cx2x);
    r[1] = vy + 2.0f * (qw * cx1y + cx2y);
    r[2] = vz + 2.0f * (qw * cx1z + cx2z);
}

__device__ inline void actSO3(const float *q, const float *X, float *Y) {
    quat_rotate_vec(q, X, Y);
}

__device__ inline void actSE3(const float *t, const float *q, const float *X, float *Y) {
    actSO3(q, X, Y);

    Y[3] = X[3];
    Y[0] += X[3] * t[0];
    Y[1] += X[3] * t[1];
    Y[2] += X[3] * t[2];
}

__device__ inline void relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
    const float qi0 = qi[0], qi1 = qi[1], qi2 = qi[2], qi3 = qi[3];
    const float qj0 = qj[0], qj1 = qj[1], qj2 = qj[2], qj3 = qj[3];
    
    // qij = qj * qi^{-1}; qi^{-1} = [-qi.xyz, qi.w]
    qij[0] = -qj3 * qi0 + qj0 * qi3 - qj1 * qi2 + qj2 * qi1;
    qij[1] = -qj3 * qi1 + qj1 * qi3 - qj2 * qi0 + qj0 * qi2;
    qij[2] = -qj3 * qi2 + qj2 * qi3 - qj0 * qi1 + qj1 * qi0;
    qij[3] = qj3 * qi3 + qj0 * qi0 + qj1 * qi1 + qj2 * qi2;

    float ti_rot[3];
    actSO3(qij, ti, ti_rot);

    tij[0] = tj[0] - ti_rot[0];
    tij[1] = tj[1] - ti_rot[1];
    tij[2] = tj[2] - ti_rot[2];
}


__global__ void fused_projective_transform_with_reduction_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> t,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> intr,
    const torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> curv,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rhs

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
        ix = static_cast<int>(ii[e]);
        jx = static_cast<int>(jj[e]);
        fx = intr[0];
        fy = intr[1];
        cx = intr[2];
        cy = intr[3];

        for (int k = 0; k < 3; k++) {
            ti[k] = t[ix][k];
            tj[k] = t[jx][k];
        }

        for (int k = 0; k < 4; k++) {
            qi[k] = q[ix][k];
            qj[k] = q[jx][k];
        }

        relSE3(ti, qi, tj, qj, tij, qij);
        if (ix == jx) {
            tij[0] = -0.1f;
            tij[1] = 0.0f;
            tij[2] = 0.0f;

            qij[0] = 0.0f;
            qij[1] = 0.0f;
            qij[2] = 0.0f;
            qij[3] = 1.0f;
        }
    }

    __syncthreads();

    for (int k = tid; k < ht * wd; k += blockDim.x) {
        const int i = k / wd;
        const int j = k % wd;
        const float u = static_cast<float>(j);
        const float v = static_cast<float>(i);

        float Xi[4];
        Xi[0] = (u - cx) / fx;
        Xi[1] = (v - cy) / fy;
        Xi[2] = 1.0f;
        Xi[3] = disps[ix][i][j];

        float Xj[4];
        actSE3(tij, qij, Xi, Xj);

        float coords_temp[2];
        float r_temp[2];

        if (Xj[2] > 0.01f) {
            const float invz = 1.0f / Xj[2];
            coords_temp[0] = fx * (Xj[0] * invz) + cx;
            coords_temp[1] = fy * (Xj[1] * invz) + cy;

        } else {
            coords_temp[0] = u;
            coords_temp[1] = v; 
        }

        r_temp[0] = target[e][i][j][0] - coords_temp[0];
        r_temp[1] = target[e][i][j][1] - coords_temp[1];

        if (Xj[2] > 0.2f && Xi[2] > 0.2f){ //is valid
            const float invz = 1.0f / Xj[2];
            const float invz2 = invz*invz;

            float Jz_temp[2];

            Jz_temp[0] = (fx * (tij[0] * invz - tij[2] * Xj[0] * invz2));
            Jz_temp[1] = (fy * (tij[1] * invz - tij[2] * Xj[1] * invz2));


            for(int ii = 0; ii <     2; ii++){
                const float w = 0.001f * weight[e][i][j][ii];
                const int iwd_j = i*wd + j;
                curv[e][iwd_j] += w * (-Jz_temp[ii]) * (-Jz_temp[ii]);
                rhs[e][iwd_j] -= w * r_temp[ii] * Jz_temp[ii];
            }

        }
    }
}


__global__ void fused_depth_jacobians_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> mono_disps,
    const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> valid_depth,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> scales,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> shifts,
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> ignore,
    const float sqrt_alpha, const float sqrt_alpha10,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> Jwq_out,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> Jd_out,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> Rd_out
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

std::tuple<torch::Tensor, torch::Tensor> fused_projective_transform_with_reduction_cuda(
    torch::Tensor t, // [E, 3]
    torch::Tensor q, // [E, 4]
    torch::Tensor disps, // [E, ht, wd]
    torch::Tensor intrinsics,
    torch::Tensor ii, // [E]
    torch::Tensor jj, // [E]
    torch::Tensor target, // [E, ht, wd, 2]
    torch::Tensor weight // [E, ht, wd, 2]
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

    torch::Tensor curv = torch::zeros({E, H*W}, opts);
    torch::Tensor rhs = torch::zeros({E, H*W}, opts);

    fused_projective_transform_with_reduction_kernel<<<E, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        target.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        weight.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        curv.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    return {curv, rhs};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_depth_jacobians_cuda(
    torch::Tensor disps,// [U, ht, wd]
    torch::Tensor mono_disps, // [U, ht, wd]
    torch::Tensor valid_depth, // [U, ht, wd]
    torch::Tensor scales, // [U]
    torch::Tensor shifts, // [U]
    torch::Tensor ignore, // [U]
    const float alpha
) {

    CHECK_INPUT(disps);
    CHECK_INPUT(mono_disps);
    CHECK_INPUTB(valid_depth);
    CHECK_INPUT(scales);
    CHECK_INPUT(shifts);
    CHECK_INPUTB(ignore);

    const int E = disps.size(0);
    const int ht = disps.size(1);
    const int wd = disps.size(2);

    const float sqrt_alpha = std::sqrt(alpha);
    const float sqrt_alpha10 = sqrt_alpha*10;

    auto opts = disps.options();

    torch::Tensor Jd = torch::zeros({E, ht*wd}, opts);
    torch::Tensor Rd = torch::zeros({E, ht*wd}, opts);
    torch::Tensor Jwq = torch::zeros({E, ht*wd, 2}, opts);

    fused_depth_jacobians_kernel<<<E, THREADS>>>(
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        mono_disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        valid_depth.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
        scales.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        shifts.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ignore.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        sqrt_alpha, sqrt_alpha10,
        Jwq.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        Jd.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        Rd.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    return {Jwq, Jd, Rd};
}
