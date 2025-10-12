#pragma once
#include "common/common.h"

Tensor quat_multiply_cuda(
    Tensor &q1,
    Tensor &q2
);

Tensor quat_rotate_cuda(
    Tensor &q,
    Tensor &points
);

Tensor quat_to_matrix_cuda(
    Tensor &q
);

Tensor matrix_to_quat_cuda(
    Tensor &R
);

std::tuple<Tensor, Tensor> se3_exp_cuda(
    Tensor &rho,
    Tensor &phi
);

std::tuple<Tensor, Tensor> se3_log_cuda(
    Tensor &t,
    Tensor &q
);

Tensor se3_point_jac_cuda(
    Tensor &p
);

Tensor se3_adjointT_cuda(
    Tensor &t,
    Tensor &q,
    Tensor &jac
);

Tensor se3_transform3d_cuda(
    Tensor &t,
    Tensor &q,
    Tensor &points
);

Tensor se3_transform4d_cuda(
    Tensor &t,
    Tensor &q,
    Tensor &points
);


