#include "common.h"
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(projective_ops_cuda, m) {
    m.doc() = "CUDA-accelerated projective operations";

    m.def("proj_cuda", &proj_cuda, py::arg("points"), py::arg("fx"),
          py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("last_dim"), "Projection");

    m.def("proj_jac_cuda", &proj_jac_cuda, py::arg("points"), py::arg("fx"),
          py::arg("fy"), py::arg("cx"), py::arg("cy"), "Jacobian of projection");
}
