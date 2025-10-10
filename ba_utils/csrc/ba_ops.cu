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

template<typename T, int N>
using PackedTensor = typename torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

template<bool ret_cross>
__global__ void fused_project_and_accumulate_kernel(
    const PackedTensor<float, 2> t,
    const PackedTensor<float, 2> q,
    const PackedTensor<float, 3> disps,
    const PackedTensor<float, 1> intr,
    const PackedTensor<long, 1> ii,
    const PackedTensor<long, 1> jj,
    const PackedTensor<float, 4> target,
    const PackedTensor<float, 4> weight,
    PackedTensor<float, 3> Hii_out,
    PackedTensor<float, 3> Hij_out,
    PackedTensor<float, 3> Hji_out,
    PackedTensor<float, 3> Hjj_out,
    PackedTensor<float, 2> vi_out,
    PackedTensor<float, 2> vj_out,
    PackedTensor<float, 3> Ei_out,
    PackedTensor<float, 3> Ej_out,
    PackedTensor<float, 2> depth_diag_out,
    PackedTensor<float, 2> depth_residual_out
){
    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    constexpr int n_warps = THREADS/32;

    const int ht = disps.size(1);
    const int wd = disps.size(2);

    __shared__ int ix;
    __shared__ int jx;
    __shared__ float fx, fy, cx, cy;
    __shared__ float ti[3], tj[3], tij[3];
    __shared__ float qi[4], qj[4], qij[4];
    __shared__ float rotT[9];

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

        create_rotT(tij, qij, rotT);
    }

    __syncthreads();

    float Hii_temp[6][6] = {0};
    float Hij_temp[6][6] = {0};
    float Hji_temp[6][6] = {0};
    float Hjj_temp[6][6] = {0};
    float vi_temp[6] = {0};
    float vj_temp[6] = {0};


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
        float Ji_temp[12];
        float Jj_temp[12];

        const float x = Xj[0];
        const float y = Xj[1];
        const float z = Xj[2];

        float invz;
        if (z > 0.01f) {
            invz = 1.0f / z;
            coords_temp[0] = fx * (x * invz) + cx;
            coords_temp[1] = fy * (y * invz) + cy;

        } else {
            coords_temp[0] = u;
            coords_temp[1] = v; 
        }

        if (z > 0.2f){ //is valid
            const float h = Xj[3];
            const float invz2 = invz*invz;
            const float fx_invz = fx*invz;
            const float fy_invz = fy*invz;
            const float neg_fx_x_invz2 = -fx*x*invz2;
            const float neg_fy_y_invz2 = -fy*y*invz2;

            // Jj computation
            Jj_temp[0] = (h*(fx_invz));
            Jj_temp[1] = (0.0f);
            Jj_temp[2] = (h*(neg_fx_x_invz2));
            Jj_temp[3] = (y*(neg_fx_x_invz2));
            Jj_temp[4] = (z*(fx_invz) - x*(neg_fx_x_invz2));
            Jj_temp[5] = (-y*(fx_invz));
            Jj_temp[6] = (0.0f);       
            Jj_temp[7] = (h*(fy_invz));
            Jj_temp[8] = (h*(neg_fy_y_invz2));
            Jj_temp[9] = (-z*(fy_invz) + y*(neg_fy_y_invz2));
            Jj_temp[10] = (-x*(neg_fy_y_invz2));
            Jj_temp[11] = (x*(fy_invz));

            // Ji computation
            neg_adjointT(rotT, tij, &Jj_temp[0], &Ji_temp[0]);

            float depth_diag = 0.0f;
            float depth_residual = 0.0f;
            float Ei_temp[6] = {0};
            float Ej_temp[6] = {0};

            for(int ii = 0; ii < 2; ii++){
                const float r = target[e][i][j][ii] - coords_temp[ii];
                const float w = 0.001f * weight[e][i][j][ii];
                const float Jz = (ii == 0) ? (fx * (tij[0] * invz - tij[2] * x * invz2))
                                           : (fy * (tij[1] * invz - tij[2] * y * invz2));
                const float wJz = w*Jz;
                
                depth_diag += (-wJz) * (-Jz);
                depth_residual += wJz * r;
                
                for(int jj = 0; jj < 6; jj++){
                    const int ii6_jj = ii*6 + jj;
                    
                    float Ji = Ji_temp[ii6_jj];
                    float Jj = Jj_temp[ii6_jj];
                    
                    if constexpr(ret_cross){
                        Ej_temp[jj] += wJz * Jj;
                        Ei_temp[jj] += wJz * Ji;
                    }

                    const float wJi = w*Ji;
                    const float wJj = w*Jj;

                    vi_temp[jj] += wJi * r;
                    vj_temp[jj] += wJj * r;

                    for(int kk = 0; kk < 6; kk++){
                        const int ii6_kk = ii*6 + kk;

                        Ji = Ji_temp[ii6_kk];
                        Jj = Jj_temp[ii6_kk];

                        Hii_temp[jj][kk] += wJi * Ji;
                        Hij_temp[jj][kk] += wJi * Jj;
                        Hji_temp[jj][kk] += wJj * Ji;
                        Hjj_temp[jj][kk] += wJj * Jj;
                    }
                }
            }
            
            const int iwd_j = i*wd + j;
            depth_diag_out[e][iwd_j] = depth_diag;
            depth_residual_out[e][iwd_j] = depth_residual;
            
            if constexpr(ret_cross){
                for(int ii = 0; ii < 6; ii++){
                    Ei_out[e][ii][iwd_j] = Ei_temp[ii];
                    Ej_out[e][ii][iwd_j] = Ej_temp[ii];
                }
            }

        }
    }

    __shared__ float vi_warp[n_warps][6];
    __shared__ float vj_warp[n_warps][6];
    __shared__ float Hii_warp[n_warps][6][6];
    __shared__ float Hij_warp[n_warps][6][6];
    __shared__ float Hji_warp[n_warps][6][6];
    __shared__ float Hjj_warp[n_warps][6][6];


    for (int i = 0; i < 6; i++){
        for (int offset = 16; offset > 0; offset/=2){
            vi_temp[i] += __shfl_down_sync(FULLMASK, vi_temp[i], offset);
            vj_temp[i] += __shfl_down_sync(FULLMASK, vj_temp[i], offset);
        }
        
        if (tid % 32 == 0){
            vi_warp[wid][i] = vi_temp[i];
            vj_warp[wid][i] = vj_temp[i];
        }

        for(int j = 0; j < 6; j++){

            for (int offset = 16; offset > 0; offset/=2){
                Hii_temp[i][j] += __shfl_down_sync(FULLMASK, Hii_temp[i][j], offset);
                Hij_temp[i][j] += __shfl_down_sync(FULLMASK, Hij_temp[i][j], offset);
                Hji_temp[i][j] += __shfl_down_sync(FULLMASK, Hji_temp[i][j], offset);
                Hjj_temp[i][j] += __shfl_down_sync(FULLMASK, Hjj_temp[i][j], offset);
            }

            if (tid % 32 == 0) {
                Hii_warp[wid][i][j] = Hii_temp[i][j];
                Hij_warp[wid][i][j] = Hij_temp[i][j];
                Hji_warp[wid][i][j] = Hji_temp[i][j];
                Hjj_warp[wid][i][j] = Hjj_temp[i][j];
            }
        }
    }

    __syncthreads();

    if (tid < 6) {
        float vi_sum_tid = 0.0f;
        float vj_sum_tid = 0.0f;

        for (int w = 0; w < n_warps; w++){
            vi_sum_tid += vi_warp[w][tid]; //parallelize over last dim
            vj_sum_tid += vj_warp[w][tid];
        }

        vi_out[e][tid] = vi_sum_tid;
        vj_out[e][tid] = vj_sum_tid;
        
        for (int i = 0; i < 6; i++){ //parallelize over second last dim
            float Hii_sum_tid_i = 0.0f;
            float Hij_sum_tid_i = 0.0f;
            float Hji_sum_tid_i = 0.0f;
            float Hjj_sum_tid_i = 0.0f;

            for (int w = 0; w < n_warps; w++){
                Hii_sum_tid_i += Hii_warp[w][tid][i];
                Hij_sum_tid_i += Hij_warp[w][tid][i];
                Hji_sum_tid_i += Hji_warp[w][tid][i];
                Hjj_sum_tid_i += Hjj_warp[w][tid][i];
            }

            Hii_out[e][tid][i] = Hii_sum_tid_i;
            Hij_out[e][tid][i] = Hij_sum_tid_i;
            Hji_out[e][tid][i] = Hji_sum_tid_i;
            Hjj_out[e][tid][i] = Hjj_sum_tid_i;
        }
    }
}

std::vector<tensor> fused_project_and_accumulate_cuda(
    tensor t, // [P, 3]
    tensor q, // [P, 4]
    tensor disps, // [E, ht, wd]
    tensor intrn, // [4] 
    tensor ii, // [E]
    tensor jj, // [E]
    tensor target, // [E, ht, wd, 2]
    tensor weight, // [E, ht, wd, 2]
    bool ret_cross12
){

    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrn);
    CHECK_INPUT(target);
    CHECK_INPUT(weight);
    CHECK_INPUTL(ii);
    CHECK_INPUTL(jj);

    const int E = ii.size(0);
    const int H = disps.size(1);
    const int W = disps.size(2);

    auto opts = disps.options();

    tensor Hii = torch::zeros({E, 6, 6}, opts);
    tensor Hij = torch::zeros({E, 6, 6}, opts);
    tensor Hji = torch::zeros({E, 6, 6}, opts);
    tensor Hjj = torch::zeros({E, 6, 6}, opts);

    tensor vi = torch::zeros({E, 6}, opts);
    tensor vj = torch::zeros({E, 6}, opts);

    tensor depth_diag = torch::zeros({E, H*W}, opts);
    tensor depth_residual = torch::zeros({E, H*W}, opts);

    if (ret_cross12){
        tensor Ei = torch::zeros({E, 6, H*W}, opts);
        tensor Ej = torch::zeros({E, 6, H*W}, opts);
        fused_project_and_accumulate_kernel<true><<<E, THREADS>>>(
            t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            intrn.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            target.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            weight.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_residual.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );

        return {Hii, Hij, Hji, Hjj, vi, vj, Ei, Ej, depth_diag, depth_residual};

    } else {
        tensor Ei = torch::empty({E, 6, H*W}, opts);
        tensor Ej = torch::empty({E, 6, H*W}, opts);
        fused_project_and_accumulate_kernel<false><<<E, THREADS>>>(
            t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            intrn.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            target.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            weight.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_residual.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );

        return {Hii, Hij, Hji, Hjj, vi, vj, depth_diag, depth_residual};
    }
}


__global__ void apply_hessian_damping_kernel(
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> hessian, // [P, P, 6, 6]
    const float ep,
    const float lm
) {
    const int p = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (p >= hessian.size(0) || tid >= 6) return;
    
    float diag_val = hessian[p][p][tid][tid];
    hessian[p][p][tid][tid] = diag_val + ep + lm * diag_val;
}

template<bool ret_cross, bool ret_depth>
__global__ void scatter_pose_system_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> Hii, // [E, 6, 6]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> Hij, // [E, 6, 6]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> Hji, // [E, 6, 6]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> Hjj, // [E, 6, 6]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> vi, // [E, 6]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> vj, // [E, 6]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> Ei,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> Ej,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> Ck, // [E, hw]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> wk, // [E, hw]
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> source_indices, // [E]
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> target_indices, // [E]
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> edge_to_keyframe,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> keyframe_indices, // [M]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> damping, // [T, ht, wd]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> hessian_out, // [P*P, 6, 6]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> gradient_out, // [P, 6]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> cross_term_out,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> depth_diag_out, // [M, hw]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> depth_gradient_out, // [M, hw]
    const int num_opt_poses,
    const int rig_size,
    const int num_fixed_poses,
    const int M,
    const int hw
) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (e >= Hii.size(0)) return;
    
    const int64_t src_frame = source_indices[e];
    const int64_t tgt_frame = target_indices[e];
    
    const int src_opt = (src_frame / rig_size) - num_fixed_poses;
    const int tgt_opt = (tgt_frame / rig_size) - num_fixed_poses;
    
    const bool src_valid = (src_opt >= 0) && (src_opt < num_opt_poses);
    const bool tgt_valid = (tgt_opt >= 0) && (tgt_opt < num_opt_poses);
    
    if (src_valid && tid < 36) {
        const int i = tid / 6;
        const int j = tid % 6;
        const int hii_idx = src_opt * num_opt_poses + src_opt;
        atomicAdd(&hessian_out[hii_idx][i][j], Hii[e][i][j]);
    }
    
    if (src_valid && tid < 6) {
        atomicAdd(&gradient_out[src_opt][tid], vi[e][tid]);
    }
    
    if (src_valid && tgt_valid && tid < 36) {
        const int i = tid / 6;
        const int j = tid % 6;
        
        const int hij_idx = src_opt * num_opt_poses + tgt_opt;
        atomicAdd(&hessian_out[hij_idx][i][j], Hij[e][i][j]);
        
        const int hji_idx = tgt_opt * num_opt_poses + src_opt;
        atomicAdd(&hessian_out[hji_idx][i][j], Hji[e][i][j]);
    }
    
    if (tgt_valid && tid < 36) {
        const int i = tid / 6;
        const int j = tid % 6;
        const int hjj_idx = tgt_opt * num_opt_poses + tgt_opt;
        atomicAdd(&hessian_out[hjj_idx][i][j], Hjj[e][i][j]);
    }
    
    if (tgt_valid && tid < 6) {
        atomicAdd(&gradient_out[tgt_opt][tid], vj[e][tid]);
    }
    
    if constexpr(ret_cross) {
        const int keyframe_idx = edge_to_keyframe[e];
        
        for (int k = tid; k < hw; k += blockDim.x) {
            if (src_valid) {
                const int cross_idx = src_opt * M + keyframe_idx;
                for (int i = 0; i < 6; i++) {
                    atomicAdd(&cross_term_out[cross_idx][i][k], Ei[e][i][k]);
                }
            }
            
            if (tgt_valid) {
                const int cross_idx = tgt_opt * M + keyframe_idx;
                for (int i = 0; i < 6; i++) {
                    atomicAdd(&cross_term_out[cross_idx][i][k], Ej[e][i][k]);
                }
            }
        }
    }
    
    if constexpr(ret_depth) {
        const int keyframe_idx = edge_to_keyframe[e];
        const int64_t frame_idx = keyframe_indices[keyframe_idx];
        
        for (int k = tid; k < hw; k += blockDim.x) {
            atomicAdd(&depth_diag_out[keyframe_idx][k], Ck[e][k]);
            atomicAdd(&depth_gradient_out[keyframe_idx][k], wk[e][k]);
        }
        
        __syncthreads();
        
        for (int k = tid; k < hw; k += blockDim.x) {
            const int i = k / (damping.size(2));
            const int j = k % (damping.size(2));
            atomicAdd(&depth_diag_out[keyframe_idx][k], damping[frame_idx][i][j]);
        }
    }
}

std::vector<torch::Tensor> scatter_pose_system_cuda(
    torch::Tensor Hii, // [E, 6, 6]
    torch::Tensor Hij, // [E, 6, 6]
    torch::Tensor Hji, // [E, 6, 6]
    torch::Tensor Hjj, // [E, 6, 6]
    torch::Tensor vi, // [E, 6]
    torch::Tensor vj, // [E, 6]
    torch::Tensor Ei, // [E, 6, hw]
    torch::Tensor Ej, // [E, 6, hw]
    torch::Tensor Ck, // [E, hw]
    torch::Tensor wk, // [E, hw]
    torch::Tensor source_indices, // [E]
    torch::Tensor target_indices, // [E]
    torch::Tensor edge_to_keyframe, // [E]
    torch::Tensor keyframe_indices, // [M]
    torch::Tensor damping, // [T, ht, wd]
    int num_opt_poses,
    int rig_size,
    int num_fixed_poses,
    bool ret_cross,
    bool ret_depth,
    int M, // num keyframes
    int ht, // height
    int wd, // width
    float ep, // epsilon damping for Hessian
    float lm // lambda (LM) damping for Hessian
) {
    const int E = Hii.size(0);
    const int manifold_dim = 6;
    const int hw = ht * wd;
    auto opts = Hii.options();
    
    const int threads = 64;
    
    tensor hessian = torch::zeros({num_opt_poses * num_opt_poses, manifold_dim, manifold_dim}, opts);
    tensor gradient = torch::zeros({num_opt_poses, manifold_dim}, opts);
    
    tensor cross_term = ret_cross ? torch::zeros({num_opt_poses * M, manifold_dim, hw}, opts)
                                  : torch::empty({1, 1, 1}, opts);
    
    tensor depth_diag = ret_depth ? torch::zeros({M, hw}, opts) : torch::empty({1, 1}, opts);
    tensor depth_gradient = ret_depth ? torch::zeros({M, hw}, opts) : torch::empty({1, 1}, opts);
    
    if (ret_cross && ret_depth) {
        scatter_pose_system_kernel<true, true><<<E, threads>>>(
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ck.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            wk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            source_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            target_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            edge_to_keyframe.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            keyframe_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            damping.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            hessian.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cross_term.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            num_opt_poses,
            rig_size,
            num_fixed_poses,
            M,
            hw
        );
    } else if (ret_cross && !ret_depth) {
        scatter_pose_system_kernel<true, false><<<E, threads>>>(
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ck.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            wk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            source_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            target_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            edge_to_keyframe.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            keyframe_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            damping.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            hessian.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cross_term.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            num_opt_poses,
            rig_size,
            num_fixed_poses,
            M,
            hw
        );
    } else if (!ret_cross && ret_depth) {
        scatter_pose_system_kernel<false, true><<<E, threads>>>(
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ck.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            wk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            source_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            target_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            edge_to_keyframe.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            keyframe_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            damping.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            hessian.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cross_term.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            num_opt_poses,
            rig_size,
            num_fixed_poses,
            M,
            hw
        );
    } else {
        scatter_pose_system_kernel<false, false><<<E, threads>>>(
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ck.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            wk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            source_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            target_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            edge_to_keyframe.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            keyframe_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            damping.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            hessian.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cross_term.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            num_opt_poses,
            rig_size,
            num_fixed_poses,
            M,
            hw
        );
    }
    
    hessian = hessian.view({num_opt_poses, num_opt_poses, manifold_dim, manifold_dim});
    
    if (ep > 0.0f || lm > 0.0f) {
        apply_hessian_damping_kernel<<<num_opt_poses, manifold_dim>>>(
            hessian.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            ep,
            lm
        );
    }
    
    if (ret_cross) {
        cross_term = cross_term.view({num_opt_poses, M, manifold_dim, hw});
    }
    
    if (ret_cross && ret_depth) {
        return {hessian, gradient, cross_term, depth_diag, depth_gradient};
    } else if (ret_cross) {
        return {hessian, gradient, cross_term};
    } else if (ret_depth) {
        return {hessian, gradient, depth_diag, depth_gradient};
    } else {
        return {hessian, gradient};
    }
}