#include "common.h"
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(projective_ops_cuda, m) {
    m.doc() = "CUDA-accelerated projective operations";

    m.def("proj_cuda", &proj_cuda, py::arg("points"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("last_dim"), "Projection");
    m.def("proj_jac_cuda", &proj_jac_cuda, py::arg("points"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"), "Jacobian of projection");

    // Fused iproj->transform->proj
    m.def("fused_projective_cuda", &fused_projective_cuda, py::arg("t"), py::arg("q"), py::arg("disps"), py::arg("intrinsics"), py::arg("ii"), py::arg("jj"), "Fused projective transform (iproj+SE3+proj)");
    m.def("fused_induced_flow_cuda", &fused_induced_flow_cuda, py::arg("t"), py::arg("q"), py::arg("disps"), py::arg("intrinsics"), py::arg("ii"), py::arg("jj"), "Fused induced flow (iproj+SE3+proj)");
    m.def("fused_projective_jac_cuda", &fused_projective_jac_cuda, py::arg("t"), py::arg("q"), py::arg("disps"), py::arg("intrinsics"), py::arg("ii"), py::arg("jj"), "Fused projective transform with jacobians (iproj+SE3+proj)");
}
