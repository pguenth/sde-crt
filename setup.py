from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import eigency

cython_path = "src/pybatch/"
include_dirs = ["src/cpp"] + eigency.get_includes()
extra_objects = ["lib/libbatch.a"]
extra_compile_args = []#['-fsanitize=address', '-fno-omit-frame-pointer']

kwargs_ext = { 'include_dirs' : include_dirs, 
               'extra_objects' : extra_objects,
               'extra_compile_args' : extra_compile_args }

extensions = [
        Extension("pybatch.pybatch", [cython_path + "pybatch.pyx"], **kwargs_ext),
        Extension("pybatch.pybreakpointstate", [cython_path + "pybreakpointstate.pyx"], **kwargs_ext),
        Extension("pybatch.breakpointstate", [cython_path + "breakpointstate.pyx"], **kwargs_ext),
        Extension("pybatch.pypseudoparticlestate", [cython_path + "pypseudoparticlestate.pyx"], **kwargs_ext)
]

dist = setup(
        name = "pybatch",
        version = "1.0",
        ext_modules = cythonize(extensions, include_path=["src"], build_dir="build/", annotate=True, language_level = 3),
        packages = ["pybatch"]
)
