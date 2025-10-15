#include "common.h"
#include <cmath>

__global__ void quat_multiply_kernel(const float *q1, const float *q2, float *result, const int batch_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int idx4 = idx * 4;

    const float q1x = q1[idx4 + 0];
    const float q1y = q1[idx4 + 1];
    const float q1z = q1[idx4 + 2];
    const float q1w = q1[idx4 + 3];

    const float q2x = q2[idx4 + 0];
    const float q2y = q2[idx4 + 1];
    const float q2z = q2[idx4 + 2];
    const float q2w = q2[idx4 + 3];

    const float rx = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y; // x
    const float ry = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x; // y
    const float rz = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w; // z
    const float rw = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z; // w
    
    const float norm = sqrtf(rx * rx + ry * ry + rz * rz + rw * rw);
    const float inv_norm = 1.0f / (norm + 1e-9f);
    
    result[idx4 + 0] = rx * inv_norm;
    result[idx4 + 1] = ry * inv_norm;
    result[idx4 + 2] = rz * inv_norm;
    result[idx4 + 3] = rw * inv_norm;
}

__global__ void single_quat_rotate_kernel(const float *q, const float *points, float *out, const int batch_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int idx3 = idx * 3;

    const float qx = q[0];
    const float qy = q[1];
    const float qz = q[2];
    const float qw = q[3];

    const float px = points[idx3 + 0];
    const float py = points[idx3 + 1];
    const float pz = points[idx3 + 2];

    const float ux = 2.0f * (qy * pz - qz * py);
    const float uy = 2.0f * (qz * px - qx * pz);
    const float uz = 2.0f * (qx * py - qy * px);

    const float vx = qy * uz - qz * uy;
    const float vy = qz * ux - qx * uz;
    const float vz = qx * uy - qy * ux;

    out[idx3 + 0] = px + qw * ux + vx;
    out[idx3 + 1] = py + qw * uy + vy;
    out[idx3 + 2] = pz + qw * uz + vz;
}

__global__ void quat_rotate_kernel(const float *q, const float *points, float *out, const int batch_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int idx3 = idx * 3;
    const int idx4 = idx * 4;

    const float qx = q[idx4 + 0];
    const float qy = q[idx4 + 1];
    const float qz = q[idx4 + 2];
    const float qw = q[idx4 + 3];

    const float px = points[idx3 + 0];
    const float py = points[idx3 + 1];
    const float pz = points[idx3 + 2];

    const float ux = 2.0f * (qy * pz - qz * py);
    const float uy = 2.0f * (qz * px - qx * pz);
    const float uz = 2.0f * (qx * py - qy * px);

    const float vx = qy * uz - qz * uy;
    const float vy = qz * ux - qx * uz;
    const float vz = qx * uy - qy * ux;

    out[idx3 + 0] = px + qw * ux + vx;
    out[idx3 + 1] = py + qw * uy + vy;
    out[idx3 + 2] = pz + qw * uz + vz;
}

__global__ void quat_to_matrix_kernel(const float *q, float *R, const int batch_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int q_idx = idx * 4;
    const int R_idx = idx * 9;

    const float qx = q[q_idx + 0];
    const float qy = q[q_idx + 1];
    const float qz = q[q_idx + 2];
    const float qw = q[q_idx + 3];

    const float qx2 = qx * qx;
    const float qy2 = qy * qy;
    const float qz2 = qz * qz;
    const float qw2 = qw * qw;

    const float qxy = qx * qy;
    const float qxz = qx * qz;
    const float qxw = qx * qw;
    const float qyz = qy * qz;
    const float qyw = qy * qw;
    const float qzw = qz * qw;

    R[R_idx + 0] = qw2 + qx2 - qy2 - qz2; // R00
    R[R_idx + 1] = 2.0f * (qxy - qzw);    // R01
    R[R_idx + 2] = 2.0f * (qxz + qyw);    // R02

    R[R_idx + 3] = 2.0f * (qxy + qzw);    // R10
    R[R_idx + 4] = qw2 - qx2 + qy2 - qz2; // R11
    R[R_idx + 5] = 2.0f * (qyz - qxw);    // R12

    R[R_idx + 6] = 2.0f * (qxz - qyw);    // R20
    R[R_idx + 7] = 2.0f * (qyz + qxw);    // R21
    R[R_idx + 8] = qw2 - qx2 - qy2 + qz2; // R22
}

__global__ void matrix_to_quat_kernel(const float *R, float *q, const int batch_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int R_idx = idx * 9;
    const int q_idx = idx * 4;

    const float R00 = R[R_idx + 0];
    const float R01 = R[R_idx + 1];
    const float R02 = R[R_idx + 2];
    const float R10 = R[R_idx + 3];
    const float R11 = R[R_idx + 4];
    const float R12 = R[R_idx + 5];
    const float R20 = R[R_idx + 6];
    const float R21 = R[R_idx + 7];
    const float R22 = R[R_idx + 8];

    const float trace = R00 + R11 + R22;

    if (trace > 0)
    {
        const float s = sqrtf(trace + 1.0f) * 2.0f;
        q[q_idx + 0] = (R21 - R12) / s;
        q[q_idx + 1] = (R02 - R20) / s;
        q[q_idx + 2] = (R10 - R01) / s;
        q[q_idx + 3] = 0.25f * s;
    }
    else if ((R00 > R11) && (R00 > R22))
    {
        const float s = sqrtf(1.0f + R00 - R11 - R22) * 2.0f;
        q[q_idx + 0] = 0.25f * s;
        q[q_idx + 1] = (R01 + R10) / s;
        q[q_idx + 2] = (R02 + R20) / s;
        q[q_idx + 3] = (R21 - R12) / s;
    }
    else if (R11 > R22)
    {
        const float s = sqrtf(1.0f + R11 - R00 - R22) * 2.0f;
        q[q_idx + 0] = (R01 + R10) / s;
        q[q_idx + 1] = 0.25f * s;
        q[q_idx + 2] = (R12 + R21) / s;
        q[q_idx + 3] = (R02 - R20) / s;
    }
    else
    {
        const float s = sqrtf(1.0f + R22 - R00 - R11) * 2.0f;
        q[q_idx + 0] = (R02 + R20) / s;
        q[q_idx + 1] = (R12 + R21) / s;
        q[q_idx + 2] = 0.25f * s;
        q[q_idx + 3] = (R10 - R01) / s;
    }
}

torch::Tensor quat_multiply_cuda(torch::Tensor &q1, torch::Tensor &q2) {
    CHECK_INPUT(q1);
    CHECK_INPUT(q2);

    const int batch_size = q1.size(0);
    auto out = torch::zeros_like(q1);

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    quat_multiply_kernel<<<blocks, THREADS>>>(q1.data_ptr<float>(), q2.data_ptr<float>(), out.data_ptr<float>(), batch_size);

    return out;
}

torch::Tensor single_quat_rotate_cuda(torch::Tensor &q, torch::Tensor &points) {
    CHECK_INPUT(q);
    CHECK_INPUT(points);

    const int batch_size = points.size(0);
    auto out = torch::zeros_like(points);

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    single_quat_rotate_kernel<<<blocks, THREADS>>>(q.data_ptr<float>(), points.data_ptr<float>(), out.data_ptr<float>(), batch_size);

    return out;
}


torch::Tensor quat_rotate_cuda(torch::Tensor &q, torch::Tensor &points) {
    CHECK_INPUT(q);
    CHECK_INPUT(points);

    const int batch_size = q.size(0);
    auto out = torch::zeros_like(points);

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    quat_rotate_kernel<<<blocks, THREADS>>>(q.data_ptr<float>(), points.data_ptr<float>(), out.data_ptr<float>(), batch_size);

    return out;
}

torch::Tensor quat_to_matrix_cuda(torch::Tensor &q) {
    CHECK_INPUT(q);

    const int batch_size = q.size(0);
    auto result = torch::zeros({batch_size, 3, 3}, q.options());

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    quat_to_matrix_kernel<<<blocks, THREADS>>>(q.data_ptr<float>(), result.data_ptr<float>(), batch_size);

    return result;
}

torch::Tensor matrix_to_quat_cuda(torch::Tensor &R) {
    CHECK_INPUT(R);

    const int batch_size = R.size(0);
    auto result = torch::zeros({batch_size, 4}, R.options());

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    matrix_to_quat_kernel<<<blocks, THREADS>>>(
        R.data_ptr<float>(),
        result.data_ptr<float>(),
        batch_size);

    return result;
}
