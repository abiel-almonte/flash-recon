#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// Common device functions for SE3/SO3 operations and transformations

/**
 * Apply SO(3) action (rotation only) to a point
 * Y = R(q) * X
 */
__device__ inline void actSO3(
    const float *q,
    const float *v,
    float *r
) {
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

/**
 * Apply SE(3) action (rotation + translation) to a point
 * Y = R(q) * X + h * t
 * where X = [x, y, z, h] with h being the homogeneous coordinate/inverse depth weight
 */
__device__ inline void actSE3(
    const float *t,
    const float *q,
    const float *X,
    float *Y
) {
    actSO3(q, X, Y);

    Y[3] = X[3];
    Y[0] += X[3] * t[0];
    Y[1] += X[3] * t[1];
    Y[2] += X[3] * t[2];
}

/**
 * Compute relative SE(3) transformation from frame i to frame j
 * T_ij = T_j * T_i^{-1}
 * Returns (t_ij, q_ij) such that p_j = T_ij * p_i
 */
__device__ inline void relSE3(
    const float *ti,
    const float *qi,
    const float *tj,
    const float *qj,
    float *tij,
    float *qij
) {
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

/**
 * Create rotation matrix (row-major, transposed) from quaternion
 * Note: rotT is the transpose of the rotation matrix R
 */
__device__ inline void create_rotT(
    const float* q,
    float* rotT
) {
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

/**
 * Compute negative adjoint transpose transformation
 * Transforms Jacobian from frame j to frame i
 * Ji = -Ad(T_ij)^T * Jj
 */
__device__ inline void neg_adjointT(
    const float* rotT,
    const float* t,
    const float* Jj,
    float* Ji_out
) { 
    const float tx = t[0];
    const float ty = t[1];
    const float tz = t[2];  
        
    for (int i = 0; i < 2; i++) {
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

/**
 * Unproject pixel coordinates to 3D point in camera frame
 * Xi = [(u - cx)/fx, (v - cy)/fy, 1, disp]
 */
__device__ inline void unproject_pixel(
    const float u,
    const float v,
    const float disp,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    float* Xi
) {
    Xi[0] = (u - cx) / fx;
    Xi[1] = (v - cy) / fy;
    Xi[2] = 1.0f;
    Xi[3] = disp;
}

/**
 * Project 3D point to pixel coordinates
 * Returns false if point is behind camera (z <= min_depth)
 */
__device__ inline bool project_point(
    const float x,
    const float y,
    const float z,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    float* u_out,
    float* v_out,
    const float min_depth = 0.01f
) {
    if (z <= min_depth) {
        return false;
    }
    const float invz = 1.0f / z;
    *u_out = fx * (x * invz) + cx;
    *v_out = fy * (y * invz) + cy;
    return true;
}

/**
 * Compute SE3 Jacobian for projective transform at frame j
 * Jj = d(proj(Xj)) / d(SE3_j) where Xj = [x, y, z, h]
 * Output: Jj is 2x6 matrix stored as 12-element array
 */
__device__ inline void compute_pose_jacobian(
    const float x,
    const float y,
    const float z,
    const float h,
    const float fx,
    const float fy,
    float* Jj_out
) {
    const float invz = 1.0f / z;
    const float invz2 = invz * invz;
    const float fx_invz = fx * invz;
    const float fy_invz = fy * invz;
    const float neg_fx_x_invz2 = -fx * x * invz2;
    const float neg_fy_y_invz2 = -fy * y * invz2;

    // Row 0 (u coordinate)
    Jj_out[0] = h * fx_invz;
    Jj_out[1] = 0.0f;
    Jj_out[2] = h * neg_fx_x_invz2;
    Jj_out[3] = y * neg_fx_x_invz2;
    Jj_out[4] = z * fx_invz - x * neg_fx_x_invz2;
    Jj_out[5] = -y * fx_invz;

    // Row 1 (v coordinate)
    Jj_out[6] = 0.0f;
    Jj_out[7] = h * fy_invz;
    Jj_out[8] = h * neg_fy_y_invz2;
    Jj_out[9] = -z * fy_invz + y * neg_fy_y_invz2;
    Jj_out[10] = -x * neg_fy_y_invz2;
    Jj_out[11] = x * fy_invz;
}

__device__ inline void compute_depth_jacobian(
    const float x,
    const float y,
    const float z,
    const float* tij,
    const float fx,
    const float fy,
    float* Jz_out
) {
    const float invz = 1.0f / z;
    const float invz2 = invz * invz;
    
    Jz_out[0] = fx * (tij[0] * invz - tij[2] * x * invz2);   
    Jz_out[1] = fy * (tij[1] * invz - tij[2] * y * invz2);
}


template<typename TensorAccessor2D, typename TensorAccessor1D, typename IndexAccessor1D>
__device__ inline void setup_projective_transform(
    const int e,
    const TensorAccessor2D& t,
    const TensorAccessor2D& q,
    const TensorAccessor1D& intr,
    const IndexAccessor1D& ii,
    const IndexAccessor1D& jj,
    int* ix,
    int* jx,
    float* fx,
    float* fy,
    float* cx,
    float* cy,
    float* ti,
    float* qi_out,
    float* tj,
    float* qj_out,
    float* tij,
    float* qij
) {
    *ix = static_cast<int>(ii[e]);
    *jx = static_cast<int>(jj[e]);
    
    *fx = intr[0];
    *fy = intr[1];
    *cx = intr[2];
    *cy = intr[3];

    for (int k = 0; k < 3; k++) {
        ti[k] = t[*ix][k];
        tj[k] = t[*jx][k];
    }

    for (int k = 0; k < 4; k++) {
        qi_out[k] = q[*ix][k];
        qj_out[k] = q[*jx][k];
    }

    relSE3(ti, qi_out,
           tj, qj_out,
           tij, qij);

    if (*ix == *jx) {
        tij[0] = -0.1f;
        tij[1] = 0.0f;
        tij[2] = 0.0f;

        qij[0] = 0.0f;
        qij[1] = 0.0f;
        qij[2] = 0.0f;
        qij[3] = 1.0f;
    }
}
