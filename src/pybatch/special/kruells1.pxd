# distutils: language = c++

from pybatch.pseudoparticlestate cimport PseudoParticleState
from pybatch.batch cimport PseudoParticleBatch
from libcpp.vector cimport vector

cdef extern from "batch_kruells1.cpp":
    pass

cdef extern from "batch_kruells1.h":
    cdef cppclass BatchKruells1(PseudoParticleBatch):
        # child methods
        BatchKruells1(double x0, double y0, int N, double Tmax, double L, double Xsh, double a, double b) except + #, double x_min, double x_max) except +
        #vector[double] integrate()

