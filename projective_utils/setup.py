from setuptools import setup, find_packages
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CUDAExtension(
        name="projective_ops_cuda",
        sources=[
            "csrc/bindings.cpp",
            "csrc/proj_ops.cu",
            "csrc/fused_proj.cu",
        ],
        include_dirs=[
            "csrc/",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
        },
    )
]

setup(
    name="projective_utils_csrc",
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
    ],
)
