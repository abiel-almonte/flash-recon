#include "common.h"
#include <tuple>
#include <cmath>

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

    if (theta < EPS)
    {
        ijac_coeff = TaylorCoeffs::IJAC_C0;
    }
    else
    {
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

__global__ void se3_point_jac_kernel(const float *p, float *j, const int n) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;

    const int idx4 = idx * 4;
    const int idx24 = idx * 24;

    const float x = p[idx4 + 0];
    const float y = p[idx4 + 1];
    const float z = p[idx4 + 2];
    const float d = p[idx4 + 3];

    // Row 1
    j[idx24] = d;
    j[idx24 + 4] = z;
    j[idx24 + 5] = -y;

    // Row 2
    j[idx24 + 7] = d;
    j[idx24 + 9] = -z;
    j[idx24 + 11] = x;

    // Row 3
    j[idx24 + 14] = d;
    j[idx24 + 15] = y;
    j[idx24 + 16] = -x;
}

template <bool HOMOG>
__global__ void se3_transform_kernel(const float *t, const float *q, const float *pts, float *out, const int B, const int S) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = B * S;

    if (idx >= N) {
        return;
    }

    const int b = idx / S;
    const int s = idx % S;
    const int D = HOMOG ? 4 : 3;
    const int base = (b * S + s) * D;

    const float vx = pts[base + 0];
    const float vy = pts[base + 1];
    const float vz = pts[base + 2];
    const float w = HOMOG ? pts[base + 3] : 1.0f;

    const float tx = t[b * 3 + 0];
    const float ty = t[b * 3 + 1];
    const float tz = t[b * 3 + 2];

    const float qx = q[b * 4 + 0];
    const float qy = q[b * 4 + 1];
    const float qz = q[b * 4 + 2];
    const float qw = q[b * 4 + 3];

    float rx, ry, rz;
    const float cx1 = qy * vz - qz * vy;
    const float cy1 = qz * vx - qx * vz;
    const float cz1 = qx * vy - qy * vx;

    const float cx2 = qy * cz1 - qz * cy1;
    const float cy2 = qz * cx1 - qx * cz1;
    const float cz2 = qx * cy1 - qy * cx1;

    rx = vx + 2.0f * (qw * cx1 + cx2);
    ry = vy + 2.0f * (qw * cy1 + cy2);
    rz = vz + 2.0f * (qw * cz1 + cz2);

    out[base + 0] = rx + tx * w;
    out[base + 1] = ry + ty * w;
    out[base + 2] = rz + tz * w;

    if (HOMOG){
        out[base + 3] = w;
    }
}

// Apply Adjoint^T to jacobians: jac [B, S, 2, 6], returns same shape
// For each row (2 of them): split [rho(3) | phi(3)]
// rho' = (rho + t x phi) * R^T
// phi' = phi * R^T
__global__ void se3_adjointT_kernel(const float* t, const float* q, const float* jac, float* out, const int B, const int S) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // over B*S*2 rows
    const int rows_per_b = S * 2;
    const int total_rows = B * rows_per_b;

    if (idx >= total_rows) {
        return;
    }

    const int b = idx / rows_per_b;
    const int rs = idx % rows_per_b; // row within [S*2]
    const int s = rs / 2;
    const int r = rs % 2; // 0 or 1

    // Load pose
    const float tx = t[b * 3 + 0];
    const float ty = t[b * 3 + 1];
    const float tz = t[b * 3 + 2];
    const float qx = q[b * 4 + 0];
    const float qy = q[b * 4 + 1];
    const float qz = q[b * 4 + 2];
    const float qw = q[b * 4 + 3];

    // Build R^T from q on the fly
    // R(q) for [qx,qy,qz,qw]
    const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const float xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const float wx = qw * qx, wy = qw * qy, wz = qw * qz;

    // R = [[1-2(yy+zz), 2(xy-wz), 2(xz+wy)],
    //      [2(xy+wz), 1-2(xx+zz), 2(yz-wx)],
    //      [2(xz-wy), 2(yz+wx), 1-2(xx+yy)]]
    // R^T = transpose(R)
    const float RT00 = 1.f - 2.f * (yy + zz);
    const float RT01 = 2.f * (xy + wz);
    const float RT02 = 2.f * (xz - wy);
    const float RT10 = 2.f * (xy - wz);
    const float RT11 = 1.f - 2.f * (xx + zz);
    const float RT12 = 2.f * (yz + wx);
    const float RT20 = 2.f * (xz + wy);
    const float RT21 = 2.f * (yz - wx);
    const float RT22 = 1.f - 2.f * (xx + yy);

    // Index into jac: [B,S,2,6]
    const int base = ((b * S + s) * 2 + r) * 6;
    float rho0 = jac[base + 0];
    float rho1 = jac[base + 1];
    float rho2 = jac[base + 2];
    float phi0 = jac[base + 3];
    float phi1 = jac[base + 4];
    float phi2 = jac[base + 5];

    // rho += t x phi
    const float cx = ty * phi2 - tz * phi1;
    const float cy = tz * phi0 - tx * phi2;
    const float cz = tx * phi1 - ty * phi0;
    rho0 += cx;
    rho1 += cy;
    rho2 += cz;

    // Right-multiply by R^T: v' = v * R^T (treat rows as row-vectors)
    const float rho0n = rho0 * RT00 + rho1 * RT01 + rho2 * RT02;
    const float rho1n = rho0 * RT10 + rho1 * RT11 + rho2 * RT12;
    const float rho2n = rho0 * RT20 + rho1 * RT21 + rho2 * RT22;
    const float phi0n = phi0 * RT00 + phi1 * RT01 + phi2 * RT02;
    const float phi1n = phi0 * RT10 + phi1 * RT11 + phi2 * RT12;
    const float phi2n = phi0 * RT20 + phi1 * RT21 + phi2 * RT22;

    out[base + 0] = rho0n;
    out[base + 1] = rho1n;
    out[base + 2] = rho2n;
    out[base + 3] = phi0n;
    out[base + 4] = phi1n;
    out[base + 5] = phi2n;
}

std::tuple<torch::Tensor, torch::Tensor> se3_exp_cuda(torch::Tensor &rho, torch::Tensor &phi) {
    CHECK_INPUT(rho);
    CHECK_INPUT(phi);

    const int batch_size = rho.size(0);

    auto q = torch::zeros({batch_size, 4}, rho.options());
    auto t = torch::zeros({batch_size, 3}, rho.options());

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    se3_exp_kernel<<<blocks, THREADS>>>(rho.data_ptr<float>(), phi.data_ptr<float>(), q.data_ptr<float>(), t.data_ptr<float>(), batch_size);

    return {t, q};
}

std::tuple<torch::Tensor, torch::Tensor> se3_log_cuda(torch::Tensor &t, torch::Tensor &q) {
    CHECK_INPUT(q);
    CHECK_INPUT(t);

    const int batch_size = q.size(0);

    auto rho = torch::zeros({batch_size, 3}, q.options());
    auto phi = torch::zeros({batch_size, 3}, q.options());

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    se3_log_kernel<<<blocks, THREADS>>>(q.data_ptr<float>(), t.data_ptr<float>(), rho.data_ptr<float>(), phi.data_ptr<float>(), batch_size);

    return {rho, phi};
}

torch::Tensor se3_point_jac_cuda(torch::Tensor &p) {
    CHECK_INPUT(p);

    const int n = p.size(0);
    auto jacobian = torch::zeros({n, 4, 6}, p.options());

    const int blocks = (n + THREADS - 1) / THREADS;

    se3_point_jac_kernel<<<blocks, THREADS>>>(p.data_ptr<float>(), jacobian.data_ptr<float>(), n);

    return jacobian;
}

torch::Tensor se3_transform3d_cuda(torch::Tensor &t, torch::Tensor &q, torch::Tensor &points) {
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(points);
    
    TORCH_CHECK(points.size(-1) == 3, "points must be [...,3]");

    const int B = t.size(0);
    const int S = points.numel() / (B * 3);

    auto out = torch::empty_like(points);

    const int blocks = (B * S + THREADS - 1) / THREADS;

    se3_transform_kernel<false><<<blocks, THREADS>>>(t.data_ptr<float>(), q.data_ptr<float>(), points.data_ptr<float>(), out.data_ptr<float>(), B, S);

    return out;
}

torch::Tensor se3_transform4d_cuda(torch::Tensor &t, torch::Tensor &q, torch::Tensor &points) {
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(points);

    TORCH_CHECK(points.size(-1) == 4, "points must be [...,4]");

    const int B = t.size(0);
    const int S = points.numel() / (B * 4);

    auto out = torch::empty_like(points);

    const int blocks = (B * S + THREADS - 1) / THREADS;

    se3_transform_kernel<true><<<blocks, THREADS>>>(t.data_ptr<float>(), q.data_ptr<float>(), points.data_ptr<float>(), out.data_ptr<float>(), B, S);

    return out;
}

torch::Tensor se3_adjointT_cuda(torch::Tensor &t, torch::Tensor &q, torch::Tensor &jac) {
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(jac);

    TORCH_CHECK(jac.size(-2) == 2 && jac.size(-1) == 6, "jac must be [...,2,6]");

    const int B = q.size(0);
    const int S = jac.numel() / (B * 2 * 6);

    auto out = torch::empty_like(jac);

    const int threads = 256;
    const int rows = B * S * 2;
    const int blocks = (rows + threads - 1) / threads;

    se3_adjointT_kernel<<<blocks, threads>>>(t.data_ptr<float>(), q.data_ptr<float>(), jac.data_ptr<float>(), out.data_ptr<float>(), B, S);

    return out;
}
