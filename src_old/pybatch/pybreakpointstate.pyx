# distutils: language = c++
# cython: language_level=3

from enum import Enum
from pybatch.breakpointstate cimport BreakpointState

class PyBreakpointState(Enum):
    UNDEFINED = 0
    NONE = 1
    UPPER = 2
    LOWER = 3
    TIME = 4

