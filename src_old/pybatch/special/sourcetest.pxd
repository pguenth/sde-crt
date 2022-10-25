# distutils: language = c++

from pybatch.pseudoparticlestate cimport PseudoParticleState
from pybatch.batch cimport PseudoParticleBatch
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string

cdef extern from "batch_sourcetest.cpp":
    pass

cdef extern from "batch_sourcetest.h":
    cdef cppclass BatchSourcetest(PseudoParticleBatch):
        # child methods
        BatchSourcetest(map[string, double] params) except +
        vector[double] integrate()

