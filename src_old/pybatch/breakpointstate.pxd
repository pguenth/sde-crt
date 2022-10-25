# distutils: language = c++
# cython: language_level=3
cdef extern from "batch/breakpointstate.h":
    ctypedef enum BreakpointState:
        UNDEFINED "BreakpointState::UNDEFINED"
        NONE "BreakpointState::NONE"
        UPPER "BreakpointState::UPPER"
        LOWER "BreakpointState::LOWER"
        TIME "BreakpointState::TIME"

cdef object py_breakpointstate(BreakpointState bp)

