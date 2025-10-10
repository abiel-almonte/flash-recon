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
#define FULLMASK 0xffffffff

using tensor = torch::Tensor;

std::tuple<torch::Tensor, torch::Tensor> fused_projective_transform_with_reduction_cuda(
    tensor t,
    tensor q,
    tensor disps,
    tensor intrinsics,
    tensor ii,
    tensor jj,
    tensor target,
    tensor weight);

std::tuple<torch::Tensor, torch::Tensor,torch::Tensor> fused_depth_jacobians_cuda(
    tensor disps,
    tensor mono_disps,
    tensor valid_depth,
    tensor scales,
    tensor shifts,
    tensor ignore,
    const float alpha);

std::vector<torch::Tensor> fused_project_and_accumulate_cuda(
    tensor t,
    tensor q,
    tensor disps,
    tensor intrn,
    tensor ii,
    tensor jj,
    tensor target,
    tensor weight,
    bool ret_cross12
);


std::vector<torch::Tensor> scatter_pose_system_cuda(
    tensor Hii,
    tensor Hij,
    tensor Hji,
    tensor Hjj,
    tensor vi,
    tensor vj,
    tensor Ei,
    tensor Ej,
    tensor Ck,
    tensor wk,
    tensor source_indices,
    tensor target_indices,
    tensor edge_to_keyframe,
    tensor keyframe_indices,
    tensor damping,
    int num_opt_poses,
    int rig_size,
    int num_fixed_poses,
    bool ret_cross,
    bool ret_depth,
    int M,
    int ht,
    int wd,
    float ep,
    float lm
);