from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため


ext = Extension(
    "sparsefuncs_fast",
    sources=["sparsefuncs_fast.pyx"],
    include_dirs=[".", get_include()],
)
setup(name="sparsefuncs_fast", ext_modules=cythonize([ext]))


