#pragma once
#include "common/common.h"
#include "common/device_ops.cuh"

template<typename T, int N>
using PackedAccessor = typename torch::PackedTensorAccessor32<T, N, at::RestrictPtrTraits>;

std::vector<Tensor> scatter_pose_system_cuda(
    Tensor Hii,
    Tensor Hij,
    Tensor Hji,
    Tensor Hjj,
    Tensor vi,
    Tensor vj,
    Tensor Ei,
    Tensor Ej,
    Tensor Ck,
    Tensor wk,
    Tensor source_indices,
    Tensor target_indices,
    Tensor edge_to_keyframe,
    Tensor keyframe_indices,
    Tensor damping,
    int num_opt_poses,
    int rig_size,
    int num_fixed_poses,
    bool ret_cross,
    bool ret_depth,
    int M,
    int ht,
    int wd,
    float ep,
    float lm
);
template <bool ret_cross, bool ret_depth>
__global__ void scatter_pose_system_kernel(
    PackedAccessor<float, 3> Hii, // [E, 6, 6]
    PackedAccessor<float, 3> Hij, // [E, 6, 6]
    PackedAccessor<float, 3> Hji, // [E, 6, 6]
    PackedAccessor<float, 3> Hjj, // [E, 6, 6]
    PackedAccessor<float, 2> vi,  // [E, 6]
    PackedAccessor<float, 2> vj,  // [E, 6]
    PackedAccessor<float, 3> Ei,
    PackedAccessor<float, 3> Ej,
    PackedAccessor<float, 2> Ck, // [E, hw]
    PackedAccessor<float, 2> wk, // [E, hw]
    PackedAccessor<long, 1> source_indices,  // [E]
    PackedAccessor<long, 1> target_indices,  // [E]
    PackedAccessor<long, 1> edge_to_keyframe,
    PackedAccessor<long, 1> keyframe_indices, // [M]
    PackedAccessor<float, 3> damping, // [T, ht, wd]
    PackedAccessor<float, 3> hessian_out, // [P*P, 6, 6]
    PackedAccessor<float, 2> gradient_out, // [P, 6]
    PackedAccessor<float, 3> cross_term_out,
    PackedAccessor<float, 2> depth_diag_out, // [M, hw]
    PackedAccessor<float, 2> depth_gradient_out, // [M, hw]
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

    if constexpr (ret_cross) {
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

    if constexpr (ret_depth) {
        const int keyframe_idx = edge_to_keyframe[e];

        for (int k = tid; k < hw; k += blockDim.x) {
            atomicAdd(&depth_diag_out[keyframe_idx][k], Ck[e][k]);
            atomicAdd(&depth_gradient_out[keyframe_idx][k], wk[e][k]);
        }

        __syncthreads();

        for (int k = tid; k < hw; k += blockDim.x) {
            const int i = k / (damping.size(2));
            const int j = k % (damping.size(2));
            atomicAdd(&depth_diag_out[keyframe_idx][k], damping[keyframe_idx][i][j]);
        }
    }
}
__global__ void apply_hessian_damping_kernel(
    PackedAccessor<float, 4> hessian, // [P, P, 6, 6]
    const float ep,
    const float lm
) {
    const int p = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (p >= hessian.size(0) || tid >= 6) return;
    
    float diag_val = hessian[p][p][tid][tid];
    hessian[p][p][tid][tid] = diag_val + ep + lm * diag_val;
}

std::vector<Tensor> fused_project_and_accumulate_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrn,
    Tensor ii,
    Tensor jj,
    Tensor target,
    Tensor weight,
    bool ret_cross12
);
template<bool ret_cross>
__global__ void fused_project_and_accumulate_kernel(
    const PackedAccessor<float, 2> t,
    const PackedAccessor<float, 2> q,
    const PackedAccessor<float, 3> disps,
    const PackedAccessor<float, 1> intr,
    const PackedAccessor<long, 1> ii,
    const PackedAccessor<long, 1> jj,
    const PackedAccessor<float, 4> target,
    const PackedAccessor<float, 4> weight,
    PackedAccessor<float, 3> Hii_out,
    PackedAccessor<float, 3> Hij_out,
    PackedAccessor<float, 3> Hji_out,
    PackedAccessor<float, 3> Hjj_out,
    PackedAccessor<float, 2> vi_out,
    PackedAccessor<float, 2> vj_out,
    PackedAccessor<float, 3> Ei_out,
    PackedAccessor<float, 3> Ej_out,
    PackedAccessor<float, 2> depth_diag_out,
    PackedAccessor<float, 2> depth_residual_out
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
        create_rotT(qij, rotT);
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
        float Ji_temp[12];
        float Jj_temp[12];

        const float x = Xj[0];
        const float y = Xj[1];
        const float z = Xj[2];

        coords_temp[0] = u;
        coords_temp[1] = v;
        float invz;
        if (project_point(
                x,
                y,
                z,
                fx,
                fy,
                cx,
                cy,
                &coords_temp[0],
                &coords_temp[1],
                0.01f
            )) {
            invz = 1.0f / z;
        }

        if (z > 0.2f){ //is valid
            const float h = Xj[3];
            const float invz2 = invz * invz;
            
            // Jj computation
            compute_pose_jacobian(
                x,
                y,
                z,
                h,
                fx,
                fy,
                Jj_temp
            );

            // Ji computation
            neg_adjointT(rotT, tij,
                        Jj_temp, Ji_temp);

            float depth_diag = 0.0f;
            float depth_residual = 0.0f;
            float Ei_temp[6] = {0};
            float Ej_temp[6] = {0};

            for(int ii = 0; ii < 2; ii++){
                const float r = target[e][i][j][ii] - coords_temp[ii];
                float w = 0.001f * weight[e][i][j][ii];
                const float Jz = (ii == 0) ? (fx * (tij[0] * invz - tij[2] * x * invz2))
                                           : (fy * (tij[1] * invz - tij[2] * y * invz2));
                const float wJz = w*Jz;
                
                depth_diag += wJz * Jz;
                depth_residual += wJz * r;
                
                if (ix == jx) w = 0.0f;
                
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
