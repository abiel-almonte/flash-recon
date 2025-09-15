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

template<bool INDUCED_FLOW>
__global__ void fused_projective_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> t,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> intr,
    const torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> valid
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

        coords[e][i][j][0] = u;
        coords[e][i][j][1] = v;
        if (Xj[2] > 0.01f) {
            const float invz = 1.0f / Xj[2];
            coords[e][i][j][0] = fx * (Xj[0] * invz) + cx;
            coords[e][i][j][1] = fy * (Xj[1] * invz) + cy;
        }

        if (INDUCED_FLOW){
            coords[e][i][j][0] -= u;
            coords[e][i][j][1] -= v;
        }
        valid[e][i][j][0] = (Xj[2] > 0.2f && Xi[2] > 0.2f) ? 1.0f : 0.0f;
    }
}

__device__ inline void create_rotT(const float* t, const float* q, float* rotT){
    const float tx = t[0];
    const float ty = t[1];
    const float tz = t[2];

    const float qx = q[0];
    const float qy = q[1];
    const float qz = q[2];
    const float qw = q[3];

    const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const float xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const float wx = qw * qx, wy = qw * qy, wz = qw * qz;

    rotT[0]= 1.0f - 2.0f * (yy + zz);
    rotT[1]= 2.0f * (xy + wz);
    rotT[2]= 2.0f * (xz - wy);
    rotT[3]= 2.0f * (xy - wz);
    rotT[4]= 1.0f - 2.0f * (xx + zz);
    rotT[5]= 2.0f * (yz + wx);
    rotT[6]= 2.0f * (xz + wy);
    rotT[7]= 2.0f * (yz - wx);
    rotT[8]= 1.0f - 2.0f * (xx + yy);
}

__device__ inline void neg_adjointT(const float* rotT, const float* t, const float* Jj, float* Ji_out) { 
    const float tx = t[0];
    const float ty = t[1];
    const float tz = t[2];  
        
    for (int i = 0; i < 2; i++){
        const int i6 = i*6;

        const float rho0 = Jj[i6 + 0];
        const float rho1 = Jj[i6 + 1];
        const float rho2 = Jj[i6 + 2];

        // phi - (t x rho)
        const float phi0 = Jj[i6 + 3] - (ty * rho2 - tz * rho1);
        const float phi1 = Jj[i6 + 4] - (tz * rho0 - tx * rho2);
        const float phi2 = Jj[i6 + 5] - (tx * rho1 - ty * rho0);
        
        // R^T * rho and R^T * phi
        for (int j = 0; j < 3; j++) {
            const int j3 = j*3;
            Ji_out[i6 + j] = -(rotT[j3 + 0]*rho0 + rotT[j3 + 1]*rho1 + rotT[j3 + 2]*rho2);
            Ji_out[i6 + j + 3] = -(rotT[j3 + 0]*phi0 + rotT[j3 + 1]*phi1 + rotT[j3 + 2]*phi2);
        }
    }
}


__global__ void fused_projective_with_cache_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> t,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> intr,
    const torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> valid,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> Xj_cache
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

        coords[e][i][j][0] = u;
        coords[e][i][j][1] = v;
        if (Xj[2] > 0.01f) {
            const float invz = 1.0f / Xj[2];
            coords[e][i][j][0] = fx * (Xj[0] * invz) + cx;
            coords[e][i][j][1] = fy * (Xj[1] * invz) + cy;
        }

        valid[e][i][j][0] = (Xj[2] > 0.2f && Xi[2] > 0.2f) ? 1.0f : 0.0f;

        // Cache transformed point for Jacobian computation
        Xj_cache[e][i][j][0] = Xj[0];
        Xj_cache[e][i][j][1] = Xj[1];
        Xj_cache[e][i][j][2] = Xj[2];
        Xj_cache[e][i][j][3] = Xj[3];
    }
}

__global__ void projective_jacobians_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> t,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> intr,
    const torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> valid,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> Xj_cache,
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> Ji_out,
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> Jj_out,
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> Jz_out
) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x;

    const int ht = disps.size(1);
    const int wd = disps.size(2);

    __shared__ int ix, jx;
    __shared__ float fx, fy, cx, cy;
    __shared__ float ti[3], tj[3], tij[3];
    __shared__ float qi[4], qj[4], qij[4];
    __shared__ float rotT[9];

    if (tid == 0) {
        ix = static_cast<int>(ii[e]);
        jx = static_cast<int>(jj[e]);
        fx = intr[0]; fy = intr[1]; cx = intr[2]; cy = intr[3];

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
        create_rotT(tij, qij, rotT);
    }
    __syncthreads();

    // Process pixels - but only compute Jacobians for valid ones
    for (int k = tid; k < ht * wd; k += blockDim.x) {
        const int i = k / wd;
        const int j = k % wd;
        
        if (valid[e][i][j][0] > 0.5f) {
            const float x = Xj_cache[e][i][j][0];
            const float y = Xj_cache[e][i][j][1]; 
            const float z = Xj_cache[e][i][j][2];
            const float h = Xj_cache[e][i][j][3];
            const float invz = 1.0f / z;

            const float fx_invz = fx*invz;
            const float fy_invz = fy*invz;
            const float neg_fx_x_invz2 = -fx*x*invz*invz;
            const float neg_fy_y_invz2 = -fy*y*invz*invz;

            // Jj computation
            Jj_out[e][i][j][0][0] = (h*(fx_invz));
            Jj_out[e][i][j][0][1] = (0.0f);
            Jj_out[e][i][j][0][2] = (h*(neg_fx_x_invz2));
            Jj_out[e][i][j][0][3] = (y*(neg_fx_x_invz2));
            Jj_out[e][i][j][0][4] = (z*(fx_invz) - x*(neg_fx_x_invz2));
            Jj_out[e][i][j][0][5] = (-y*(fx_invz));

            Jj_out[e][i][j][1][0] = (0.0f);       
            Jj_out[e][i][j][1][1] = (h*(fy_invz));
            Jj_out[e][i][j][1][2] = (h*(neg_fy_y_invz2));
            Jj_out[e][i][j][1][3] = (-z*(fy_invz) + y*(neg_fy_y_invz2));
            Jj_out[e][i][j][1][4] = (-x*(neg_fy_y_invz2));
            Jj_out[e][i][j][1][5] = (x*(fy_invz));

            // Jz computation
            Jz_out[e][i][j][0][0] = (fx * (tij[0] * invz - tij[2] * x * invz * invz));
            Jz_out[e][i][j][1][0] = (fy * (tij[1] * invz - tij[2] * y * invz * invz));

            // Ji computation
            neg_adjointT(rotT, tij, &Jj_out[e][i][j][0][0], &Ji_out[e][i][j][0][0]);
        }
    }
}

std::vector<torch::Tensor> fused_projective_cuda(
    torch::Tensor t,          // [B,3]
    torch::Tensor q,          // [B,4]
    torch::Tensor disps,      // [B,H,W]
    torch::Tensor intrinsics, // [4]
    torch::Tensor ii, torch::Tensor jj
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
    torch::Tensor coords = torch::zeros({E, H, W, 2}, opts);
    torch::Tensor valid = torch::zeros({E, H, W, 1}, opts);

    fused_projective_kernel<false><<<E, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        valid.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );

    return {coords, valid};
}


std::vector<torch::Tensor> fused_induced_flow_cuda(
    torch::Tensor t,          // [B,3]
    torch::Tensor q,          // [B,4]
    torch::Tensor disps,      // [B,H,W]
    torch::Tensor intrinsics, // [4]
    torch::Tensor ii, torch::Tensor jj
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
    torch::Tensor coords = torch::zeros({E, H, W, 2}, opts);
    torch::Tensor valid = torch::zeros({E, H, W, 1}, opts);

    fused_projective_kernel<true><<<E, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        valid.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
    );

    return {coords, valid};
}



std::vector<torch::Tensor> fused_projective_jac_cuda(
    torch::Tensor t,
    torch::Tensor q,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj
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
    auto half_opts = opts.dtype(torch::kFloat16);

    torch::Tensor coords = torch::zeros({E, H, W, 2}, opts);
    torch::Tensor valid = torch::zeros({E, H, W, 1}, opts);
    torch::Tensor Xj_cache = torch::zeros({E, H, W, 4}, opts);
    
    torch::Tensor Ji = torch::zeros({E, H, W, 2, 6}, opts);
    torch::Tensor Jj = torch::zeros({E, H, W, 2, 6}, opts);
    torch::Tensor Jz = torch::zeros({E, H, W, 2, 1}, opts);

    fused_projective_with_cache_kernel<<<E, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        valid.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        Xj_cache.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

    projective_jacobians_kernel<<<E, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        valid.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        Xj_cache.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        Ji.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        Jj.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        Jz.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

    return {coords, valid, Ji, Jj, Jz};
}
