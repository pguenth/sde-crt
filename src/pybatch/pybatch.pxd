# distutils: language = c++

from pybatch.batch cimport PseudoParticleBatch

from pybatch.pseudoparticlestate cimport PseudoParticleState
from pybatch.pypseudoparticlestate cimport PyPseudoParticleState

from eigency.core cimport *
from libcpp.vector cimport vector
from libcpp.map cimport map as cpp_map
from libcpp.string cimport string
from libcpp cimport bool

import numpy as np
cimport numpy as np

#https://github.com/cython/cython/wiki/WrappingSetOfCppClasses
cdef cpp_map[string, double] dict_to_map_string_double(dict d)
#cdef cpp_map[string, double] dict_to_map_string_double(np.ndarray[str] k, np.ndarray[np.float64_t] v)

cdef class PyPseudoParticleBatch:
    cdef PseudoParticleBatch *_batch
    cdef dict _params
    cdef bool _reconstructed
    cdef list _states
    cdef list _integrator_values
