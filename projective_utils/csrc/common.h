#pragma once

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::kLong, #x " must be long")

#define CHECK_INPUT(x)   \
    CHECK_DEVICE(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_FLOAT32(x)
#define CHECK_INPUTL(x)  \
    CHECK_DEVICE(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_LONG(x)

#define THREADS 256
#define HALF_MINDEPTH 0.1

torch::Tensor proj_jac_cuda(torch::Tensor &p, const float fx, const float fy, const float cx, const float cy);
torch::Tensor proj_cuda(torch::Tensor &p, const float fx, const float fy, const float cx, const float cy, const int last_dim);

// Fused iproj -> transform -> proj
std::vector<torch::Tensor> fused_projective_cuda(
    torch::Tensor t,
    torch::Tensor q,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj);

std::vector<torch::Tensor> fused_projective_jac_cuda(
    torch::Tensor t,
    torch::Tensor q,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj);
