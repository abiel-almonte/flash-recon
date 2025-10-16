#include "kernels.h"

PYBIND11_MODULE(corr, m){
    m.def(
        "corr_forward",
        &corr_forward, 
        py::arg("volume"), py::arg("coords"), py::arg("radius"),
        "Compute correlations"
    );
}