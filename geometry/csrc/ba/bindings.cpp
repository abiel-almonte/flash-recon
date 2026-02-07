#include "kernels.h"

PYBIND11_MODULE(ba, m) {
    m.doc() = "CUDA-accelerated bundle adjustment operations";
    
    m.def(
        "fused_projective_transform_with_reduction_cuda", 
        &fused_projective_transform_with_reduction_cuda, 
        py::arg("t"), 
        py::arg("q"),
        py::arg("disps"),
        py::arg("intrinsics"), 
        py::arg("ii"),
        py::arg("jj"), 
        py::arg("target"),
        py::arg("weight"),
        "Fused projective transform with reduction"
    );

    m.def(
        "fused_depth_jacobians_cuda",
        &fused_depth_jacobians_cuda, 
        py::arg("disps"),
        py::arg("mono_depths"),
        py::arg("valid_depth"),
        py::arg("scales"),
        py::arg("shifts"),
        py::arg("ignore"),
        py::arg("alpha"),
        "Fused operation to compute depth jacobians"
    );

    m.def(
        "fused_project_and_accumulate_cuda",
        &fused_project_and_accumulate_cuda,
        py::arg("t"), 
        py::arg("q"), 
        py::arg("disps"), 
        py::arg("intrinsics"), 
        py::arg("ii"), 
        py::arg("jj"), 
        py::arg("target"), 
        py::arg("weight"), 
        py::arg("ret_cross12"),
        "Projective transform and assemble linear system"
    );

    m.def(
        "scatter_pose_system_cuda",
        &scatter_pose_system_cuda,
        py::arg("Hii"), 
        py::arg("Hij"), 
        py::arg("Hji"), 
        py::arg("Hjj"), 
        py::arg("vi"), 
        py::arg("vj"), 
        py::arg("Ei"), 
        py::arg("Ej"),
        py::arg("Ck"), 
        py::arg("wk"),
        py::arg("source_indices"), 
        py::arg("target_indices"), 
        py::arg("edge_to_keyframe"),
        py::arg("keyframe_indices"), 
        py::arg("damping"),
        py::arg("num_opt_poses"), 
        py::arg("rig_size"), 
        py::arg("num_fixed_poses"),
        py::arg("ret_cross"), 
        py::arg("ret_depth"), 
        py::arg("M"), 
        py::arg("ht"), 
        py::arg("wd"),
        py::arg("ep"), 
        py::arg("lm"),
        "Fused scatter of per-edge Hessians/gradients, cross-terms, and depth terms into global BA system with optional damping"
    );
}
