from setuptools import setup
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CUDAExtension(
        name='projective_ops_cuda',
        sources=[
            'csrc/bindings.cpp',
            'csrc/proj_ops.cu',
        ],
        include_dirs=[
            'csrc/',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math', '-lineinfo']
        }
    )
]

setup(
    name='projective_utils',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=['src'],
    package_dir={'src': 'src'},
    install_requires=[
        'torch',
    ],
)
