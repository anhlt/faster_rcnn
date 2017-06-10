from distutils.core import setup
import numpy as np
from distutils.extension import Extension
from Cython.Distutils import build_ext


try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "cython_bbox",
        ["bbox.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
        include_dirs=[numpy_include]
    ),
    Extension(
        "cython_nms",
        ["nms.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
        include_dirs=[numpy_include]
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
