# distutils: language = c++

from libcpp cimport bool
from eigency.core cimport *
from cscheme cimport *


cdef class PySpaceTimePoint:
    cdef SpaceTimePoint *_spacetimepoint
    cdef bool _owned

    @staticmethod
    cdef object _from_cptr(SpaceTimePoint *ptr, bool owned)
