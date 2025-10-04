#include "common.h"
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(ba_ops_cuda, m) {
    m.doc() = "CUDA-accelerated bundle adjustment operations";
    
    m.def("fused_projective_transform_with_reduction_cuda", &fused_projective_transform_with_reduction_cuda, 
        py::arg("t"), py::arg("q"), py::arg("disps"), py::arg("intrinsics"), py::arg("ii"), py::arg("jj"), py::arg("target"), py::arg("weight"),
         "Fused projective transform with reduction (iproj+SE3+proj)");
        
    m.def("fused_depth_jacobians_cuda", &fused_depth_jacobians_cuda, 
        py::arg("disps"), py::arg("mono_disps"), py::arg("valid_depth"), py::arg("scales"), py::arg("shifts"), py::arg("ignore"), py::arg("alpha"),
         "Fused operation to compute depth jacobians");
}   
