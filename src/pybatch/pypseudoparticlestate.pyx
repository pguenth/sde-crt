# distutils: language = c++

from pybatch.pseudoparticlestate cimport PseudoParticleState, SpaceTimePoint
from pybatch.breakpointstate cimport py_breakpointstate
from eigency.core cimport *
from libcpp.vector cimport vector

from pybatch.pybreakpointstate import *
import numpy as np
from libcpp cimport bool

cdef class PySpaceTimePoint:
    #cdef SpaceTimePoint *_spacetimepoint
    #cdef bool _owned
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

    @staticmethod
    cdef object from_cobj(PseudoParticleState c):
        cdef PyPseudoParticleState w = PyPseudoParticleState.__new__(PyPseudoParticleState)
        w._ppstate = c

        return w

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
        return self._ppstate.finished()

    @property
    def breakpoint_state(self):
        return py_breakpointstate(self._ppstate.get_breakpoint_state())

#    def __str__(self):
#        return str(self._ppstate)


