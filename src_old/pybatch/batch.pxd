# distutils: language = c++

from pybatch.pseudoparticlestate cimport PseudoParticleState
from libcpp.vector cimport vector

cdef extern from "batch/pseudoparticlebatch.cpp":
    pass

cdef extern from "batch/pseudoparticlebatch.h":
    cdef cppclass PseudoParticleBatch:
        int run(int particle_count, int nthreads)
        int step_all(int steps)
        int unfinished_count()

        PseudoParticleState& state(int index)
        const vector[PseudoParticleState] states()
        vector[vector[double]] get_integrator_values()
