#include "common.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(lie_ops_cuda, m) {
    m.doc() = "CUDA-accelerated Lie group operations for SO(3) and SE(3)";

    // SO(3) operations
    m.def("quat_multiply_cuda", &quat_multiply_cuda, py::arg("q1"), py::arg("q2"), "Quaternion multiplication");
    m.def("quat_rotate_cuda", &quat_rotate_cuda, py::arg("q"), py::arg("points"), "Rotate points by quaternion");
    m.def("single_quat_rotate_cuda", &single_quat_rotate_cuda, py::arg("q"), py::arg("points"), "Rotate points by quaternion");
    m.def("quat_to_matrix_cuda", &quat_to_matrix_cuda, py::arg("q"), "Convert quaternion to rotation matrix");
    m.def("matrix_to_quat_cuda", &matrix_to_quat_cuda, py::arg("R"), "Convert rotation matrix to quaternion");

    // SE(3) operations
    m.def("se3_exp_cuda", &se3_exp_cuda, py::arg("rho"), py::arg("phi"), "SE3 exponential map");
    m.def("se3_log_cuda", &se3_log_cuda, py::arg("t"), py::arg("q"), "SE3 logarithm map");
    m.def("se3_point_jac_cuda", &se3_point_jac_cuda, py::arg("p"), "SE3 point jacobian");

    // Fused transforms
    m.def("se3_transform3d_cuda", &se3_transform3d_cuda, py::arg("t"), py::arg("q"), py::arg("points"), "Transform 3D points by SE3");
    m.def("se3_transform4d_cuda", &se3_transform4d_cuda, py::arg("t"), py::arg("q"), py::arg("points"), "Transform 4D homogeneous points by SE3");
    m.def("se3_adjointT_cuda", &se3_adjointT_cuda, py::arg("t"), py::arg("q"), py::arg("jac"), "Apply SE3 adjoint transpose to jacobians");
}
