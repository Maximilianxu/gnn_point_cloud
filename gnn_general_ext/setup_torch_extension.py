from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gnn_ext_forward',
    ext_modules=[
        CUDAExtension(
          name='gnn_ext_forward', 
          sources=[   'src/main.cpp',
                      'src/forward.cpp', 
                      'src/cuda/sage.cu'
                  ],
          include_dirs=["include/"]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
