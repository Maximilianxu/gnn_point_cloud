from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ExtForward',
    ext_modules=[
        CUDAExtension(
          name='ext_forward', 
          sources=[   
                      'forward.cpp', 
                      'sage.cu'
                  ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })