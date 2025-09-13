#include "common.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(lie_ops_cuda, m) {
    m.doc() = "CUDA-accelerated Lie group operations for SO(3) and SE(3)";

    // SO(3) operations
    m.def("quat_multiply_cuda", &quat_multiply_cuda, "Quaternion multiplication");
    m.def("quat_rotate_cuda", &quat_rotate_cuda, "Rotate points by quaternion");
    m.def("quat_to_matrix_cuda", &quat_to_matrix_cuda, "Convert quaternion to rotation matrix");
    m.def("matrix_to_quat_cuda", &matrix_to_quat_cuda, "Convert rotation matrix to quaternion");

    // SE(3) operations
    m.def("se3_exp_cuda", &se3_exp_cuda, "SE3 exponential map");
    m.def("se3_log_cuda", &se3_log_cuda, "SE3 logarithm map");
    m.def("se3_point_jac_cuda", &se3_point_jac_cuda, "SE3 point jacobian");
}
