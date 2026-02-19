import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT = os.path.dirname(os.path.abspath(__file__))
common_includes = [os.path.join(ROOT, "csrc"), os.path.join(ROOT, "csrc/common")]


corr_module = CUDAExtension(
    name=f"neural_cuda.corr",
    sources=[
        f"csrc/corr/bindings.cpp",
        f"csrc/corr/kernels.cu",
    ],
    include_dirs=[os.path.join(ROOT, f"csrc/corr"), *common_includes],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
    },
)


setup(
    name="neural",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[corr_module],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
