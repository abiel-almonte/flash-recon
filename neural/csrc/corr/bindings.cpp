#include "kernels.h"

PYBIND11_MODULE(corr, m){
    m.def(
        "corr_forward",
        &corr_forward, 
        py::arg("volume"), py::arg("coords"), py::arg("radius"),
        "Compute correlations"
    );
    m.def(
        "altcorr_forward",
        &altcorr_forward, 
        py::arg("fmap1"), py::arg("fmap2"), py::arg("coords"), py::arg("radius"),
        "Compute correlations on the fly"
    );
}
