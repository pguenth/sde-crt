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
        BatchKruells921(double x0, double y0, int N, double Tmax, double Tesc) except + #, double x_min, double x_max) except +
        #vector[double] integrate()

    cdef cppclass BatchKruells922(PseudoParticleBatch):
        # child methods
        BatchKruells922(double x0, double y0, int N, double Tmax, double dxs, double Kpar, double r, double Vs, double dt, double beta_s) except + #, double x_min, double x_max) except +
        #vector[double] integrate()

    cdef cppclass BatchKruells923(PseudoParticleBatch):
        # child methods
        BatchKruells923(double x0, double y0, double r_inj, double Tmax, double dxs, double Kpar, double r, double Vs, double dt, double beta_s) except + #, double x_min, double x_max) except +
        #vector[double] integrate()

    cdef cppclass BatchKruells924(PseudoParticleBatch):
        # child methods
        BatchKruells924(map[string, double] params) except +
        #vector[double] integrate()
