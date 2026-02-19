import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT = os.path.dirname(os.path.abspath(__file__))
common_includes = [os.path.join(ROOT, "csrc"), os.path.join(ROOT, "csrc/common")]


def create_module(name):
    return CUDAExtension(
        name=f"geometry_cuda.{name}",
        sources=[
            f"csrc/{name}/bindings.cpp",
            f"csrc/{name}/kernels.cu",
        ],
        include_dirs=[os.path.join(ROOT, f"csrc/{name}"), *common_includes],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
        },
    )


setup(
    name="geometry",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[create_module(m) for m in ["ba", "proj", "lie"]],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
