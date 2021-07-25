# distutils: language = c++

from pybatch.pseudoparticlestate cimport PseudoParticleState
from pybatch.batch cimport PseudoParticleBatch
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string

cdef extern from "batch_kruells92.cpp":
    pass

cdef extern from "batch_kruells92.h":
    cdef cppclass BatchKruells921(PseudoParticleBatch):
        # child methods
        BatchKruells921(map[string, double] params) except + #, double x_min, double x_max) except +
        #vector[double] integrate()

    cdef cppclass BatchKruells922(PseudoParticleBatch):
        # child methods
        BatchKruells922(map[string, double] params) except + #, double x_min, double x_max) except +
        #vector[double] integrate()

    cdef cppclass BatchKruells923(PseudoParticleBatch):
        # child methods
        BatchKruells923(map[string, double] params) except + #, double x_min, double x_max) except +
        #vector[double] integrate()

    cdef cppclass BatchKruells924(PseudoParticleBatch):
        # child methods
        BatchKruells924(map[string, double] params) except +
        #vector[double] integrate()

    cdef cppclass BatchKruells925(PseudoParticleBatch):
        # child methods
        BatchKruells925(map[string, double] params) except +
        #vector[double] integrate()
