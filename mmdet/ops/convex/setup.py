  
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='riroi_align_cuda',
    ext_modules=[
        CUDAExtension('convex_ext', [
            'src/convex_cpu.cpp',
            'src/convex_ext.cpp',
        ],
        sources_cuda=['src/convex_cuda.cu']
        ),
    ],
    cmdclass={'build_ext': BuildExtension})
