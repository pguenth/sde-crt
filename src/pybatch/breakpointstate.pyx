# distutils: language = c++
# cython: language_level=3

from pybatch.breakpointstate cimport BreakpointState
from pybatch.pybreakpointstate import PyBreakpointState

cdef object py_breakpointstate(BreakpointState bp):
    return PyBreakpointState(<int> bp)

# not implemented, not sure how to do it effectively
#cdef c_breakpointstate(bp):
#   pass
