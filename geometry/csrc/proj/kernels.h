#pragma once
#include "common/common.h"
#include <vector>

Tensor proj_cuda(
    Tensor &p,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const int last_dim
);

Tensor proj_jac_cuda(
    Tensor &p,
    const float fx,
    const float fy,
    const float cx,
    const float cy
);

std::vector<Tensor> fused_projective_jac_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj
);

std::vector<Tensor> fused_projective_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj
);

std::vector<Tensor> fused_induced_flow_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj
);

Tensor fused_depth_filter_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor thresh
);

Tensor frame_distance_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj,
    float beta
);
