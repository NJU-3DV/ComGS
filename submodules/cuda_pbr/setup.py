

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="cuda_pbr",
    packages=['cuda_pbr'],
    ext_modules=[
        CUDAExtension(
            name="cuda_pbr._C",
            sources=[
            "src/render_equation.cu",
            "src/d2n.cu",
            "src/sample.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
