#pragma once
#include "common/common.h"

struct TaylorCoeffs {
    static constexpr float IMAG_C0 = 0.5f;
    static constexpr float IMAG_C2 = -1.0f / 48.0f;
    static constexpr float IMAG_C4 = 1.0f / 3840.0f;

    static constexpr float REAL_C0 = 1.0f;
    static constexpr float REAL_C2 = -1.0f / 8.0f;
    static constexpr float REAL_C4 = 1.0f / 384.0f;

    static constexpr float JAC_C0 = 1.0f / 2.0f;
    static constexpr float JAC_C1 = 1.0f / 24.0f;
    static constexpr float JAC_C2 = 1.0f / 6.0f;
    static constexpr float JAC_C3 = 1.0f / 120.0f;

    static constexpr float ATAN_C0 = 2.0f / 3.0f;

    static constexpr float IJAC_C0 = -1.0f / 12.0f;
};

std::tuple<Tensor, Tensor> se3_exp_cuda(
    Tensor &rho,
    Tensor &phi
);
__global__ void se3_exp_kernel(const float *rho, const float *phi, float *q, float *t, const int batch_size) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int idx3 = idx * 3;
    const int idx4 = idx * 4;

    const float phi_x = phi[idx3 + 0];
    const float phi_y = phi[idx3 + 1];
    const float phi_z = phi[idx3 + 2];

    const float theta2 = phi_x * phi_x + phi_y * phi_y + phi_z * phi_z;
    const float theta = sqrt(theta2);

    float imag_factor = 0.0f;
    float real_factor = 0.0f;
    float jac_coeff1 = 0.0f;
    float jac_coeff2 = 0.0f;

    if (theta < EPS)
    {
        const float theta4 = theta2 * theta2;
        imag_factor = TaylorCoeffs::IMAG_C0 + TaylorCoeffs::IMAG_C2 * theta2 + TaylorCoeffs::IMAG_C4 * theta4;
        real_factor = TaylorCoeffs::REAL_C0 + TaylorCoeffs::REAL_C2 * theta2 + TaylorCoeffs::REAL_C4 * theta4;
        jac_coeff1 = TaylorCoeffs::JAC_C0 - TaylorCoeffs::JAC_C1 * theta2;
        jac_coeff2 = TaylorCoeffs::JAC_C2 - TaylorCoeffs::JAC_C3 * theta2;
    }
    else
    {
        imag_factor = sinf(0.5f * theta) / theta;
        real_factor = cosf(0.5f * theta);
        jac_coeff1 = (1.0f - cosf(theta)) / theta2;
        jac_coeff2 = (theta - sinf(theta)) / (theta * theta2);
    }

    // Compute quaternion (SO3 exponential)
    q[idx4 + 0] = imag_factor * phi_x;
    q[idx4 + 1] = imag_factor * phi_y;
    q[idx4 + 2] = imag_factor * phi_z;
    q[idx4 + 3] = real_factor;

    // Precompute squared terms for Jacobian matrix
    const float phi_x2 = phi_x * phi_x;
    const float phi_y2 = phi_y * phi_y;
    const float phi_z2 = phi_z * phi_z;

    // Precompute cross terms for Jacobian matrix
    const float phi_xy = phi_x * phi_y;
    const float phi_xz = phi_x * phi_z;
    const float phi_yz = phi_y * phi_z;

    const float rho_x = rho[idx3 + 0];
    const float rho_y = rho[idx3 + 1];
    const float rho_z = rho[idx3 + 2];

    // Compute translation t = J(phi) * rho using precomputed terms
    t[idx3 + 0] = rho_x * (1.0f + jac_coeff2 * (-phi_y2 - phi_z2)) +
                  rho_y * (-jac_coeff1 * phi_z + jac_coeff2 * phi_xy) +
                  rho_z * (jac_coeff1 * phi_y + jac_coeff2 * phi_xz);

    t[idx3 + 1] = rho_x * (jac_coeff1 * phi_z + jac_coeff2 * phi_xy) +
                  rho_y * (1.0f + jac_coeff2 * (-phi_x2 - phi_z2)) +
                  rho_z * (-jac_coeff1 * phi_x + jac_coeff2 * phi_yz);

    t[idx3 + 2] = rho_x * (-jac_coeff1 * phi_y + jac_coeff2 * phi_xz) +
                  rho_y * (jac_coeff1 * phi_x + jac_coeff2 * phi_yz) +
                  rho_z * (1.0f + jac_coeff2 * (-phi_x2 - phi_y2));
}

std::tuple<Tensor, Tensor> se3_log_cuda(
    Tensor &t,
    Tensor &q
);
__global__ void se3_log_kernel(const float *q, const float *t, float *rho, float *phi, const int batch_size) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int idx3 = idx * 3;
    const int idx4 = idx * 4;

    const float qx = q[idx4 + 0];
    const float qy = q[idx4 + 1];
    const float qz = q[idx4 + 2];
    const float qw = q[idx4 + 3];

    const float n2 = qx * qx + qy * qy + qz * qz;
    float atan_coeff = 0.0f;

    if (n2 < EPS * EPS)
    {
        atan_coeff = (2.0f / qw) - TaylorCoeffs::ATAN_C0 * (n2 / (qw * qw * qw));
    }
    else
    {
        const float n = std::sqrt(n2);

        if (std::abs(qw) < EPS)
        {
            if (qw > 0.0f)
            {
                atan_coeff = PI / n;
            }
            else
            {
                atan_coeff = -PI / n;
            }
        }
        else
        {
            atan_coeff = 2.0f * std::atan(n / qw) / n;
        }
    }

    // Compute phi (SO3 log)
    const float phi_x = atan_coeff * qx;
    const float phi_y = atan_coeff * qy;
    const float phi_z = atan_coeff * qz;

    phi[idx3 + 0] = phi_x;
    phi[idx3 + 1] = phi_y;
    phi[idx3 + 2] = phi_z;

    // Precompute squared terms for Jacobian inv matrix
    const float phi_x2 = phi_x * phi_x;
    const float phi_y2 = phi_y * phi_y;
    const float phi_z2 = phi_z * phi_z;

    // Precompute cross terms for Jacobian inv matrix
    const float phi_xy = phi_x * phi_y;
    const float phi_xz = phi_x * phi_z;
    const float phi_yz = phi_y * phi_z;

    const float theta2 = phi_x * phi_x + phi_y * phi_y + phi_z * phi_z;
    const float theta = sqrt(theta2);
    const float half_theta = theta / 2.0f;

    float ijac_coeff = 0.0f;

    if (theta < EPS){
        ijac_coeff = TaylorCoeffs::IJAC_C0;
    }
    else{
        ijac_coeff = (1.0f - theta * (cosf(half_theta) / (2.0f * sinf(half_theta)))) / (theta2);
    }

    const float tx = t[idx3 + 0];
    const float ty = t[idx3 + 1];
    const float tz = t[idx3 + 2];

    // V^(-1) = I - 0.5*[phi]_x + coef2*[phi]_x^2
    rho[idx3 + 0] = tx * (1.0f + ijac_coeff * (-phi_y2 - phi_z2)) +
                    ty * (0.5f * phi_z + ijac_coeff * phi_xy) +
                    tz * (-0.5f * phi_y + ijac_coeff * phi_xz);

    rho[idx3 + 1] = tx * (-0.5f * phi_z + ijac_coeff * phi_xy) +
                    ty * (1.0f + ijac_coeff * (-phi_x2 - phi_z2)) +
                    tz * (0.5f * phi_x + ijac_coeff * phi_yz);

    rho[idx3 + 2] = tx * (0.5f * phi_y + ijac_coeff * phi_xz) +
                    ty * (-0.5f * phi_x + ijac_coeff * phi_yz) +
                    tz * (1.0f + ijac_coeff * (-phi_x2 - phi_y2));
}
