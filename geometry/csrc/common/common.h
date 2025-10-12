#pragma once

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define THREADS 256
#define FULLMASK 0xffffffff
#define EPS 1e-6
#define PI 3.14159265358979323846
#define HALF_MINDEPTH 0.1

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


using Tensor = torch::Tensor;
