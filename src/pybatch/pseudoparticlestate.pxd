# distutils: language = c++

from eigency.core cimport *
from libcpp.vector cimport vector
from libcpp cimport bool

from pybatch.breakpointstate cimport BreakpointState

cdef extern from "batch/pseudoparticlestate.cpp":
    pass

cdef extern from "batch/pseudoparticlestate.h":
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

        #writing state
        void update(double t, Map[VectorXd]& x)
        void finish(BreakpointState b)

        bool finished()
        BreakpointState get_breakpoint_state()
        #const vector[double] get_integrator_values()

        PseudoParticleState()

        #operator std::string()

