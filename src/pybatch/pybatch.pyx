# distutils: language = c++

from pybatch.batch cimport PseudoParticleBatch

from pybatch.pseudoparticlestate cimport PseudoParticleState
from pybatch.pypseudoparticlestate cimport PyPseudoParticleState

from eigency.core cimport *
from libcpp.vector cimport vector

import numpy as np

#https://github.com/cython/cython/wiki/WrappingSetOfCppClasses

cdef class PyPseudoParticleBatch:
    #cdef PseudoParticleBatch *_batch

    def run(self, int particle_count=-1):
        return (<PseudoParticleBatch *>(self._batch)).run(particle_count)

    def step_all(self, int steps=1):
        return (<PseudoParticleBatch *>(self._batch)).step_all(steps)

    @property
    def unfinished_count(self):
        return (<PseudoParticleBatch *>(self._batch)).unfinished_count()

    def state(self, int index):
        cdef PseudoParticleState s = (<PseudoParticleBatch *>(self._batch)).state(index)
        return PyPseudoParticleState.from_cobj(s)

    def states(self):
        rlist = []
        cdef vector[PseudoParticleState] vlist = (<PseudoParticleBatch *>(self._batch)).states()
        for s in vlist:
            rlist.append(PyPseudoParticleState.from_cobj(s))

        return rlist

    def __dealloc__(self):
        if not self._batch is NULL:
            del self._batch
            self._batch = NULL

