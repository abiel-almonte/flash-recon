#pragma once
#include "common/common.h"
#include "common/device_ops.cuh"

template<typename T, int N>
using PackedAccessor = typename torch::PackedTensorAccessor32<T, N, at::RestrictPtrTraits>;

std::vector<Tensor> fused_projective_jac_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj
);
__global__ void fused_projective_with_cache_kernel(
    const PackedAccessor<float, 2> t,
    const PackedAccessor<float, 2> q,
    const PackedAccessor<float, 3> disps,
    const PackedAccessor<float, 1> intr,
    const PackedAccessor<long, 1> ii,
    const PackedAccessor<long, 1> jj,
    PackedAccessor<float, 4> coords,
    PackedAccessor<float, 4> valid,
    PackedAccessor<float, 4> Xj_cache
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

        coords[e][i][j][0] = u;
        coords[e][i][j][1] = v;
        project_point(
            Xj[0],
            Xj[1],
            Xj[2],
            fx,
            fy,
            cx,
            cy,
            &coords[e][i][j][0],
            &coords[e][i][j][1],
            0.01f
        );

        valid[e][i][j][0] = (Xj[2] > 0.2f && Xi[2] > 0.2f) ? 1.0f : 0.0f;

        // Cache transformed point for Jacobian computation
        Xj_cache[e][i][j][0] = Xj[0];
        Xj_cache[e][i][j][1] = Xj[1];
        Xj_cache[e][i][j][2] = Xj[2];
        Xj_cache[e][i][j][3] = Xj[3];
    }
}

__global__ void projective_jacobians_kernel(
    const PackedAccessor<float, 2> t,
    const PackedAccessor<float, 2> q,
    const PackedAccessor<float, 3> disps,
    const PackedAccessor<float, 1> intr,
    const PackedAccessor<long, 1> ii,
    const PackedAccessor<long, 1> jj,
    const PackedAccessor<float, 4> valid,
    const PackedAccessor<float, 4> Xj_cache,
    PackedAccessor<float, 5> Ji_out,
    PackedAccessor<float, 5> Jj_out,
    PackedAccessor<float, 5> Jz_out
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

    for (int k = tid; k < ht * wd; k += blockDim.x) {
        const int i = k / wd;
        const int j = k % wd;

        if (valid[e][i][j][0] > 0.5f) {
            const float x = Xj_cache[e][i][j][0];
            const float y = Xj_cache[e][i][j][1];
            const float z = Xj_cache[e][i][j][2];
            const float h = Xj_cache[e][i][j][3];

            // Jj computation
            float Jj_temp[12];
            compute_pose_jacobian(
                x,
                y,
                z,
                h,
                fx,
                fy,
                Jj_temp
            );
            for (int jj = 0; jj < 6; jj++) {
                Jj_out[e][i][j][0][jj] = Jj_temp[jj];
                Jj_out[e][i][j][1][jj] = Jj_temp[6 + jj];
            }

            // Jz computation
            float Jz_temp[2];
            compute_depth_jacobian(
                x,
                y,
                z,
                tij,
                fx,
                fy,
                Jz_temp
            );
            Jz_out[e][i][j][0][0] = Jz_temp[0];
            Jz_out[e][i][j][1][0] = Jz_temp[1];

            // Ji computation
            neg_adjointT(rotT, tij,
                        Jj_temp, &Ji_out[e][i][j][0][0]);
        }
    }
}

std::vector<Tensor> fused_projective_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj
);
std::vector<Tensor> fused_induced_flow_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj
);
template<bool INDUCED_FLOW>
__global__ void fused_projective_kernel(
    const PackedAccessor<float, 2> t,
    const PackedAccessor<float, 2> q,
    const PackedAccessor<float, 3> disps,
    const PackedAccessor<float, 1> intr,
    const PackedAccessor<long, 1> ii,
    const PackedAccessor<long, 1> jj,
    PackedAccessor<float, 4> coords,
    PackedAccessor<float, 4> valid
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

        coords[e][i][j][0] = u;
        coords[e][i][j][1] = v;
        project_point(
            Xj[0],
            Xj[1],
            Xj[2],
            fx,
            fy,
            cx,
            cy,
            &coords[e][i][j][0],
            &coords[e][i][j][1],
            0.01f
        );

        if (INDUCED_FLOW) {
            coords[e][i][j][0] -= u;
            coords[e][i][j][1] -= v;
        }
        valid[e][i][j][0] = (Xj[2] > 0.2f && Xi[2] > 0.2f) ? 1.0f : 0.0f;
    }
}


Tensor fused_depth_filter_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor thresh
);
__global__ void depth_filter_kernel(
    const PackedAccessor<float, 2> t,
    const PackedAccessor<float, 2> q,
    const PackedAccessor<float, 3> disps,
    const PackedAccessor<float, 1> intr,
    const PackedAccessor<long, 1> ii,
    const PackedAccessor<float, 1> thresh,
    PackedAccessor<float, 3> counter_out
) {
	const int m = blockIdx.x;
	const int neigh_id = blockIdx.y;
	const int tid = threadIdx.x;
	const int idx = blockIdx.z * blockDim.x + tid;

    const int n = disps.size(0);
    const int ht = disps.size(1);
    const int wd = disps.size(2);

    const float threshold = thresh[m];

    __shared__ int ix;
    __shared__ int jx;
    __shared__ float fx, fy, cx, cy;
    __shared__ float ti[3], tj[3], tij[3];
    __shared__ float qi[4], qj[4], qij[4];


    if (tid == 0) {
        ix = static_cast<int>(ii[m]);
        jx = (neigh_id < 3) ? ix - neigh_id - 1 : ix + neigh_id - 2; // droid-slam impl: ix - neigh_id - 1 : ix + neigh_id 
        fx = intr[0];
        fy = intr[1];
        cx = intr[2];
        cy = intr[3];
    }
    __syncthreads();


    if (jx < 0 || jx >= n){
        return;
    }
    
    if (tid == 0){
        for (int k = 0; k < 3; k++) {
            ti[k] = t[ix][k];
            tj[k] = t[jx][k];
        }
    
        for (int k = 0; k < 4; k++) {
            qi[k] = q[ix][k];
            qj[k] = q[jx][k];
        }
    
        relSE3(ti, qi,
               tj, qj,
               tij, qij);
    }
    __syncthreads();

    
	if (idx < ht*wd) {
        const int i = idx / wd;
        const int j = idx % wd;
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
        
        const float invz = 1.0f / (Xj[2] + EPS);

        const float uj = fx * (Xj[0] * invz) + cx;
        const float vj = fy * (Xj[1] * invz) + cy;
        const float invdj = 1.0f / (Xj[3] * invz + EPS);

        const int u0 = __float2int_rd(uj);
        const int v0 = __float2int_rd(vj);

        if (u0 >= 0 && v0 >= 0 && u0 < wd-1 && v0 < ht-1) {
            const float invd00 = 1.0f / (disps[jx][v0+0][u0+0] + EPS);
            const float invd01 = 1.0f / (disps[jx][v0+0][u0+1] + EPS);
            const float invd10 = 1.0f / (disps[jx][v0+1][u0+0] + EPS);
            const float invd11 = 1.0f / (disps[jx][v0+1][u0+1] + EPS);

            if (abs(invdj - invd00) < threshold) { 
                atomicAdd(&counter_out[m][i][j], 1.0f);
            } else if  (abs(invdj - invd01) < threshold) {
                atomicAdd(&counter_out[m][i][j], 1.0f);
            } else if  (abs(invdj - invd10) < threshold) {
                atomicAdd(&counter_out[m][i][j], 1.0f);
            } else if  (abs(invdj - invd11) < threshold) {
                atomicAdd(&counter_out[m][i][j], 1.0f);
            }
        }
    }
}
