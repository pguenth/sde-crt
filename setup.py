from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import eigency
import numpy

kwargs_ext = { 'include_dirs' : eigency.get_includes() + [numpy.get_include(), 'sdesolver/loop/', '/usr/include/eigen3'],
               #'library_dirs' : ["lib/"],
               #'libraries' : ["batch"], # dynamic linking at link time
               #'extra_objects' : ["lib/libbatch.a"], #static linking
               'extra_compile_args' : ["-std=c++11", "-O3"]#['-fsanitize=address', '-fno-omit-frame-pointer']
             }

extensions = [
        Extension("sdesolver.loop.pyloop", ["sdesolver/loop/pyloop.pyx"], **kwargs_ext),
        Extension("sdesolver.util.cprint", ["sdesolver/util/cprint.pyx"], **kwargs_ext)
]

dist = setup(
        name = "sdesolver",
        version = "1.0",
        ext_modules = cythonize(extensions, build_dir="build/", annotate=True, language_level = 3),
        packages = ["sdesolver"]
)
