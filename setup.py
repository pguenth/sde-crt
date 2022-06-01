from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import eigency

cython_path = "src/pybatch/"

kwargs_ext = { 'include_dirs' : ["src/cpp"] + eigency.get_includes(),
               'library_dirs' : ["lib/"],
               'libraries' : ["batch"], # dynamic linking at link time
               #'extra_objects' : ["lib/libbatch.a"], #static linking
               'extra_compile_args' : ["-std=c++11"]#['-fsanitize=address', '-fno-omit-frame-pointer']
             }

extensions = [
        Extension("pybatch.pybatch", [cython_path + "pybatch.pyx"], **kwargs_ext),
        Extension("pybatch.pybreakpointstate", [cython_path + "pybreakpointstate.pyx"], **kwargs_ext),
        Extension("pybatch.breakpointstate", [cython_path + "breakpointstate.pyx"], **kwargs_ext),
        Extension("pybatch.pypseudoparticlestate", [cython_path + "pypseudoparticlestate.pyx"], **kwargs_ext)
]

special = [
        Extension("pybatch.special.kruells", [cython_path + "special/kruells.pyx"], **kwargs_ext),
        Extension("pybatch.special.sourcetest", [cython_path + "special/sourcetest.pyx"], **kwargs_ext)
]

dist = setup(
        name = "pybatch",
        version = "1.0",
        ext_modules = cythonize(extensions + special, include_path=["src"], build_dir="build/", annotate=True, language_level = 3),
        packages = ["pybatch"]
)
