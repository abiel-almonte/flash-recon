#pragma once
#include "common/common.h"


std::tuple<Tensor, Tensor> fused_projective_transform_with_reduction_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrinsics,
    Tensor ii,
    Tensor jj,
    Tensor target,
    Tensor weight
);

std::tuple<Tensor, Tensor, Tensor> fused_depth_jacobians_cuda(
    Tensor disps,
    Tensor mono_depths,
    Tensor valid_depth,
    Tensor scales,
    Tensor shifts,
    Tensor ignore,
    const float alpha
);

std::vector<Tensor> fused_project_and_accumulate_cuda(
    Tensor t,
    Tensor q,
    Tensor disps,
    Tensor intrn,
    Tensor ii,
    Tensor jj,
    Tensor target,
    Tensor weight,
    bool ret_cross12
);

std::vector<Tensor> scatter_pose_system_cuda(
    Tensor Hii,
    Tensor Hij,
    Tensor Hji,
    Tensor Hjj,
    Tensor vi,
    Tensor vj,
    Tensor Ei,
    Tensor Ej,
    Tensor Ck,
    Tensor wk,
    Tensor source_indices,
    Tensor target_indices,
    Tensor edge_to_keyframe,
    Tensor keyframe_indices,
    Tensor damping,
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


