# distutils: language = c++

from pybatch.batch cimport PseudoParticleBatch

from pybatch.pseudoparticlestate cimport PseudoParticleState
from pybatch.pypseudoparticlestate cimport PyPseudoParticleState

from eigency.core cimport *
from libcpp.vector cimport vector

import numpy as np

#https://github.com/cython/cython/wiki/WrappingSetOfCppClasses

cdef class PyPseudoParticleBatch:
    cdef PseudoParticleBatch *_batch
