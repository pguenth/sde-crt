# distutils: language = c++

from pybatch.pseudoparticlestate cimport PseudoParticleState
from libcpp.vector cimport vector

cdef extern from "pseudoparticlebatch.cpp":
    pass

cdef extern from "pseudoparticlebatch.h":
    cdef cppclass PseudoParticleBatch:
        int run(int particle_count)
        int step_all(int steps)
        int unfinished_count()

        PseudoParticleState& state(int index)
        const vector[PseudoParticleState] states()

cdef extern from "batch.cpp":
    pass

cdef extern from "batch.h":
    cdef cppclass BatchSourcetest(PseudoParticleBatch):
        # inherited from PseudoParticleBatch

        # child methods
        BatchSourcetest(double x0, int N, double Tmax, double x_min, double x_max) except +
        vector[double] integrate()

