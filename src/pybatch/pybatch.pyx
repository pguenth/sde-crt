# distutils: language = c++

from pybatch.batch cimport BatchSourcetest
from pybatch.batch cimport PseudoParticleBatch

from pybatch.pseudoparticlestate cimport PseudoParticleState
from pybatch.pypseudoparticlestate cimport PyPseudoParticleState

from eigency.core cimport *
from libcpp.vector cimport vector

import numpy as np
#from pybatch.pypseudoparticlestate import *
#from pybatch.pybreakpointstate import *

#https://github.com/cython/cython/wiki/WrappingSetOfCppClasses

cdef class PyPseudoParticleBatch:
    cdef PseudoParticleBatch *_batch

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

cdef class PyBatchSourcetest(PyPseudoParticleBatch):
    def __cinit__(self, double x0, int N, double Tmax, double x_min, double x_max):
        self._batch = <PseudoParticleBatch *>(new BatchSourcetest(x0, N, Tmax, x_min, x_max))

    @staticmethod
    cdef BatchSourcetest *_cast_(PseudoParticleBatch *_ptr):
        return <BatchSourcetest *>_ptr

    def __dealloc__(self):
        cdef BatchSourcetest *tmp
        if not self._batch is NULL:
            tmp = <BatchSourcetest *>self._batch;
            del tmp
            self._batch = NULL

    def integrate(self):
        return (<BatchSourcetest *>self._batch).integrate()
