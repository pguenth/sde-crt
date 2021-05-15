# distutils: language = c++

from pybatch.pseudoparticlestate cimport PseudoParticleState
from pybatch.batch cimport PseudoParticleBatch
from libcpp.vector cimport vector

cdef extern from "batch_sourcetest.cpp":
    pass

cdef extern from "batch_sourcetest.h":
    cdef cppclass BatchSourcetest(PseudoParticleBatch):
        # child methods
        BatchSourcetest(double x0, int N, double Tmax, double x_min, double x_max) except +
        vector[double] integrate()

