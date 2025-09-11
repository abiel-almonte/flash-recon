#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

#define EPS 1e-6
#define PI 3.14159265358979323846

torch::Tensor quat_multiply_cuda(torch::Tensor q1, torch::Tensor q2);
torch::Tensor quat_rotate_cuda(torch::Tensor q, torch::Tensor points);
torch::Tensor quat_to_matrix_cuda(torch::Tensor q);
torch::Tensor matrix_to_quat_cuda(torch::Tensor R);

std::tuple<torch::Tensor, torch::Tensor> se3_exp_cuda(torch::Tensor& rho, torch::Tensor& phi);
torch::Tensor se3_log_cuda(torch::Tensor& t, torch::Tensor& q);
torch::Tensor se3_point_jac_cuda(torch::Tensor& p, const int n);