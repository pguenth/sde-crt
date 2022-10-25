from pybatch.pseudoparticlestate cimport PseudoParticleState, SpaceTimePoint
from libcpp cimport bool

cdef class PyPseudoParticleState:
    cdef PseudoParticleState _ppstate
    cdef object _finished
    cdef object _breakpointstate

    @staticmethod
    cdef object from_cobj(PseudoParticleState c)

cdef class PySpaceTimePoint:
    cdef SpaceTimePoint *_spacetimepoint
    cdef bool _owned

    @staticmethod
    cdef object _from_cptr(SpaceTimePoint *ptr, bool owned)
