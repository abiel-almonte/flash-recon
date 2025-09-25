#pragma once

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::kLong, #x " must be long")
#define CHECK_BOOL(x) TORCH_CHECK(x.scalar_type() == at::kBool, #x " must be bool")

#define CHECK_INPUT(x)   \
    CHECK_DEVICE(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_FLOAT32(x)
#define CHECK_INPUTL(x)  \
    CHECK_DEVICE(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_LONG(x)
#define CHECK_INPUTB(x)  \
    CHECK_DEVICE(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_BOOL(x)

#define THREADS 256

std::tuple<torch::Tensor, torch::Tensor> fused_projective_transform_with_reduction_cuda(
    torch::Tensor t,
    torch::Tensor q,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor target,
    torch::Tensor weight);

std::tuple<torch::Tensor, torch::Tensor,torch::Tensor> fused_depth_jacobians_cuda(
    torch::Tensor disps,
    torch::Tensor mono_disps,
    torch::Tensor valid_depth,
    torch::Tensor scales,
    torch::Tensor shifts,
    torch::Tensor ignore,
    const float alpha);