from libcpp cimport bool
from eigency.core cimport *
from libcpp.vector cimport vector
from libcpp.string cimport string

#cdef extern from "cscheme.cpp":
#    pass

cdef extern from "scheme.h":
    ctypedef void (*coeff_call_t)(double *out, double t, const double *x)


cdef extern from "loop.h":
    #ctypedef VectorXd (*drift_t)(const SpaceTimePoint&)
    #ctypedef MatrixXd (*diffusion_t)(const SpaceTimePoint&)
    ctypedef void (*rng_call_t)(VectorXd& x_out, int ndim); 
    ctypedef int (*boundary_call_t)(double t, const VectorXd& x);

    cdef void ploop (vector[VectorXd]& observations, double t0, const Map[VectorXd]& x0,
                      coeff_call_t drift, coeff_call_t diffusion, boundary_call_t boundary,
                      rng_call_t rng, double timestep, vector[double]& t_observe, string scheme_name)

    cdef void ploop_pointer (double *observations, double t0, const Map[VectorXd]& x0,
                      coeff_call_t drift, coeff_call_t diffusion, boundary_call_t boundary,
                      rng_call_t rng, double timestep, const double *t_observe, int t_observe_count, string scheme_name)
