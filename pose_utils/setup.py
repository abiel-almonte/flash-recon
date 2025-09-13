from setuptools import setup, find_packages
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CUDAExtension(
        name="lie_ops_cuda",
        sources=[
            "csrc/bindings.cpp",
            "csrc/so3_ops.cu",
            "csrc/se3_ops.cu",
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
    name="pose_utils_csrc",
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
    ],
)
