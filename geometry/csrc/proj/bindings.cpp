#include "kernels.h"

PYBIND11_MODULE(proj, m) {
    m.doc() = "CUDA-accelerated projective operations";

    m.def(
        "proj_cuda", 
        &proj_cuda, 
        py::arg("points"), 
        py::arg("fx"), 
        py::arg("fy"), 
        py::arg("cx"), 
        py::arg("cy"), 
        py::arg("last_dim"),
        "Projection"
    );

    m.def(
        "proj_jac_cuda", 
        &proj_jac_cuda, 
        py::arg("points"), 
        py::arg("fx"), 
        py::arg("fy"), 
        py::arg("cx"), 
        py::arg("cy"),
        "Jacobian of projection"
    );

    m.def(
        "fused_projective_cuda", 
        &fused_projective_cuda,
        py::arg("t"),
        py::arg("q"),
        py::arg("disps"),
        py::arg("intrinsics"),
        py::arg("ii"),
        py::arg("jj"),
        "Fused projective transform (iproj+SE3+proj)"
    );

    m.def(
        "fused_induced_flow_cuda",
        &fused_induced_flow_cuda,
        py::arg("t"),
        py::arg("q"),
        py::arg("disps"),
        py::arg("intrinsics"),
        py::arg("ii"),
        py::arg("jj"),
        "Fused induced flow"
    );

    m.def(
        "fused_projective_jac_cuda",
        &fused_projective_jac_cuda,
        py::arg("t"),
        py::arg("q"),
        py::arg("disps"),
        py::arg("intrinsics"),
        py::arg("ii"),
        py::arg("jj"),
        "Fused projective transform with jacobians"
    );

    m.def(
        "fused_depth_filter_cuda",
        &fused_depth_filter_cuda,
        py::arg("t"),
        py::arg("q"),
        py::arg("disps"),
        py::arg("intrinsics"),
        py::arg("ii"),
        py::arg("thresh"),
        "Count number of neighboring depths are consistent"
    );

    m.def(
        "frame_distance_cuda",
        &frame_distance_cuda,
        py::arg("t"),
        py::arg("q"),
        py::arg("disps"),
        py::arg("intrinsics"),
        py::arg("ii"),
        py::arg("jj"),
        py::arg("beta"),
        "Compute reprojection-based frame distance"
    );
}
