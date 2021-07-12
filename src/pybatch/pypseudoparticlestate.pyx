# distutils: language = c++

from pybatch.pseudoparticlestate cimport PseudoParticleState, SpaceTimePoint
from pybatch.breakpointstate cimport py_breakpointstate
from eigency.core cimport *
from libcpp.vector cimport vector

from pybatch.pybreakpointstate import *
import numpy as np
from libcpp cimport bool

cdef class PySpaceTimePoint:
    def __cinit__(self, double t, np.ndarray[np.float64_t] x):
        self._spacetimepoint = new SpaceTimePoint(t, Map[VectorXd](x))
        self._owned = True

    @staticmethod
    cdef object _from_cptr(SpaceTimePoint *ptr, bool owned):
        cdef PySpaceTimePoint stp = PySpaceTimePoint.__new__(PySpaceTimePoint)
        stp._spacetimepoint = ptr
        stp._owned = owned
        return stp

    def __dealloc__(self):
        if self._owned:
            del self._spacetimepoint

    def __reduce__(self):
        return type(self)._reconstruct, (self.t, self.x)
    
    @classmethod
    def _reconstruct(cls, t, x):
        return cls(t, x)

    @property
    def t(self):
        return self._spacetimepoint.t

    @property
    def x(self):
        return ndarray(self._spacetimepoint.x).T[0]

cdef class PyPseudoParticleState:
    #cdef PseudoParticleState _ppstate
    def __cinit__(self):
        self._ppstate = PseudoParticleState()
        self._finished = None
        self._breakpointstate = None

    @staticmethod
    cdef object from_cobj(PseudoParticleState c):
        cdef PyPseudoParticleState w = PyPseudoParticleState.__new__(PyPseudoParticleState)
        w._ppstate = c

        return w

    def __reduce__(self):
        return type(self)._reconstruct, (self.trajectory, self.finished, self.breakpoint_state)

    @classmethod
    def _reconstruct(cls, trajectory, finished, bpstate):
        instance = cls()

        for p in trajectory:
            instance.update(p)

        if finished:
            instance.finish(bpstate)

        return instance

    @property
    def x(self):
        return ndarray(self._ppstate.get_x())

    @property
    def t(self):
        return self._ppstate.get_t()

    @property
    def p(self):
        cdef double test = self._ppstate.get_t()
        cdef VectorXd test2 = self._ppstate.get_x()
        return PySpaceTimePoint(test, ndarray(test2).T[0])

    @property
    def trajectory(self):
        cdef vector[SpaceTimePoint] v = self._ppstate.get_trajectory()
        python_list = []
        for item in v:
            python_list.append(PySpaceTimePoint(item.t, ndarray(item.x).T[0]))

        return python_list

    @property
    def finished(self):
        if self._finished is None:
            self._finished = self._ppstate.finished()
        return self._finished

    @property
    def breakpoint_state(self):
        if self._breakpointstate is None:
            self._breakpointstate = py_breakpointstate(self._ppstate.get_breakpoint_state())
        return self._breakpointstate

    def update(self, p):
        self._ppstate.update(p.t, Map[VectorXd](p.x))

    def finish(self, b):
        #self._ppstate.finish(c_breakpointstate(b))
        self._finished = True
        self._breakpointstate = b

#    def __str__(self):
#        return str(self._ppstate)


