from libcpp cimport bool
from eigency.core cimport *

#cdef extern from "cscheme.cpp":
#    pass

cdef extern from "cscheme.h":
    cdef cppclass SpaceTimePoint:
        double t
        VectorXd x

        SpaceTimePoint(double t, Map[VectorXd]& x) except +
        SpaceTimePoint() except +
        #operator std::string()

    ctypedef VectorXd (*drift_t)(const SpaceTimePoint&)
    ctypedef MatrixXd (*diffusion_t)(const SpaceTimePoint&)
    ctypedef void (*coeff_call_t)(double *out, double t, const double *x)

    cpdef SpaceTimePoint scheme_euler(const SpaceTimePoint& p, const Map[VectorXd]& rndvec, double timestep, coeff_call_t drift, coeff_call_t diffusion) 
