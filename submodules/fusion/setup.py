
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="fusion",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="fusion._C",
            sources=[
                "csrc/fusion.cpp",
                "csrc/ext.cpp"
            ])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
