# distutils: language = c++
# cython: language_level=3

from enum import Enum
from pybatch.breakpointstate cimport BreakpointState

class PyBreakpointState(Enum):
    UNDEFINED = -1
    NONE = 0
    UPPER = 1
    LOWER = 2
    TIME = 3

