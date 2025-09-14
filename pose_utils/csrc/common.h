#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float")

#define CHECK_INPUT(x)   \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x); \
    CHECK_FLOAT(x)

#define THREADS 256
#define EPS 1e-6
#define PI 3.14159265358979323846

torch::Tensor quat_multiply_cuda(torch::Tensor &q1, torch::Tensor &q2);
torch::Tensor quat_rotate_cuda(torch::Tensor &q, torch::Tensor &points);
torch::Tensor single_quat_rotate_cuda(torch::Tensor &q, torch::Tensor &points);
torch::Tensor quat_to_matrix_cuda(torch::Tensor &q);
torch::Tensor matrix_to_quat_cuda(torch::Tensor &R);

std::tuple<torch::Tensor, torch::Tensor> se3_exp_cuda(torch::Tensor &rho, torch::Tensor &phi);
std::tuple<torch::Tensor, torch::Tensor> se3_log_cuda(torch::Tensor &t, torch::Tensor &q);
torch::Tensor se3_point_jac_cuda(torch::Tensor &p);

// Fused SE3 transforms (3D / 4D)
torch::Tensor se3_transform3d_cuda(torch::Tensor &t, torch::Tensor &q, torch::Tensor &points);
torch::Tensor se3_transform4d_cuda(torch::Tensor &t, torch::Tensor &q, torch::Tensor &points);

// Fused adjoint transpose application: jac shape [B, S, 2, 6]
torch::Tensor se3_adjointT_cuda(torch::Tensor &t, torch::Tensor &q, torch::Tensor &jac);