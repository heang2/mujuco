"""
Build script for Cython extensions.

Usage:
    cd src/cython
    python setup.py build_ext --inplace

Or from the project root:
    python src/cython/setup.py build_ext --inplace --build-lib .
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="fast_gae",
        sources=["fast_gae.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    name="mujuco_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        annotate=True,   # generates .html annotation for profiling
    ),
    include_dirs=[np.get_include()],
)
