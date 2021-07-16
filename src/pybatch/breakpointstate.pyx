# distutils: language = c++
# cython: language_level=3

from pybatch.breakpointstate cimport BreakpointState
from pybatch.pybreakpointstate import PyBreakpointState

cdef object py_breakpointstate(BreakpointState bp):
    #print(bp)
    return PyBreakpointState(<int> bp)

cdef BreakpointState c_breakpointstate(bp):
    if bp == PyBreakpointState.UNDEFINED:
        return BreakpointState.UNDEFINED
    elif bp == PyBreakpointState.NONE:
        return BreakpointState.NONE
    elif bp == PyBreakpointState.UPPER:
        return BreakpointState.UPPER
    elif bp == PyBreakpointState.LOWER:
        return BreakpointState.LOWER
    elif bp == PyBreakpointState.TIME:
        return BreakpointState.TIME


