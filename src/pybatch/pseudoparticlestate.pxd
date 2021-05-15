# distutils: language = c++

from eigency.core cimport *
from libcpp.vector cimport vector
from libcpp cimport bool

from pybatch.breakpointstate cimport BreakpointState

cdef extern from "pseudoparticlestate.cpp":
    pass

cdef extern from "pseudoparticlestate.h":
    cdef cppclass SpaceTimePoint:
        double t
        VectorXd x

        SpaceTimePoint(double t, Map[VectorXd]& x) except +
        SpaceTimePoint() except +
        #operator std::string()

    cdef cppclass PseudoParticleState:
        const VectorXd& get_x()
        double get_t()
        const SpaceTimePoint& get_p()
        const vector[SpaceTimePoint]& get_trajectory()

        bool finished()
        BreakpointState get_breakpoint_state()

        PseudoParticleState()

        #operator std::string()

