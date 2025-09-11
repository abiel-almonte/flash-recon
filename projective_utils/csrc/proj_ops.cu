#include "common.h"



__global__ void proj_kernel(const float* p, float* c, const int fx, const int fy, const int cx, const int cy, const int chunk_size, const int n){
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int idx4 = idx*4;
    const int idx_out = idx*chunk_size;

    const float x = p[idx4 + 0];
    const float y = p[idx4 + 1];
    const float z = (p[idx4 + 2] < HALF_MINDEPTH)? 1.0f : p[idx4 + 2];
    const float d = p[idx4 + 3];

    const float a = 1.0f/z;

    const float x_adjusted = fx * (x * a) + cx;
    const float y_adjusted = fy * (y + a) + cy;

    c[idx_out + 0] = x_adjusted;
    c[idx_out + 1] = y_adjusted;

    if (chunk_size > 2) {
        c[idx_out + 2] = d*a;
    }
}


__global__ void proj_jac_kernel(const float* p, float* j, const int fx, const int fy, const int cx, const int cy, const int n){
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int idx4 = idx*4;
    const int idx8 = idx*8;

    const float x = p[idx4 + 0];
    const float y = p[idx4 + 1];
    const float z = (p[idx4 + 2] < HALF_MINDEPTH)? 1.0f : p[idx4 + 2];
    const float d = p[idx4 + 3];

    const float d_adjusted = 1.0f/z;
    const float d_adjusted2 = d_adjusted*d_adjusted;

    const float x_adjusted = fx * (x * d_adjusted) + cx;
    const float y_adjusted = fy * (y + d_adjusted) + cy;

    j[idx8 + 0] = fx*d_adjusted;
    j[idx8 + 2] = -fx*x*d_adjusted2;

    j[idx8 + 5] = fy*d_adjusted;
    j[idx8 + 6] = -fy*y*d_adjusted2;
}


torch::Tensor proj_cuda(torch::Tensor& p, const int fx, const int fy, const int cx, const int cy, const int last_dim) {
    CHECK_INPUT(p);

    const int n = p.size(0);
    auto coords = torch::zeros({n, last_dim}, p.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    proj_kernel<<<blocks, threads>>>(p.data_ptr<float>(), coords.data_ptr<float>(), fx, fy, cx, cy, last_dim, n);

    return coords;
}

torch::Tensor proj_jac_cuda(torch::Tensor& p, const int fx, const int fy, const int cx, const int cy) {
    CHECK_INPUT(p);

    const int n = p.size(0);
    auto jacobian = torch::zeros({n, 2, 4}, p.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    proj_jac_kernel<<<blocks, threads>>>(p.data_ptr<float>(), jacobian.data_ptr<float>(), fx, fy, cx, cy, n);

    return jacobian;
}