from libcpp cimport bool
from eigency.core cimport *
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "scheme.h":
    ctypedef void (*coeff_call_t)(double *out, double t, const double *x)


cdef extern from "loop.h":
    ctypedef void (*rng_call_t)(VectorXd& x_out, int ndim); 
    ctypedef int (*boundary_call_t)(double t, const VectorXd& x);
    ctypedef bool (*split_call_t)(double t, const VectorXd& x, double last_t, const VectorXd& last_x);

    # unused in cython
    # call signature probably wrong
    cdef int integration_loop (vector[VectorXd]& observations, double *t, Map[VectorXd]& x,
                     coeff_call_t drift, coeff_call_t diffusion, boundary_call_t boundary, 
                     split_call_t split, long seed, # rng_call_t rng,
                     double timestep, vector[double]& t_observe, int *split_count, 
                     vector[VectorXd]& split_times, vector[VectorXd]& split_points, vector[VectorXd]& split_weights, vector[VectorXd]& this_weights, double initial_weight, string scheme_name) nogil

    cdef int integration_loop_p (double *observations, int *observation_count, double *t, Map[VectorXd]& x,
                      coeff_call_t drift, coeff_call_t diffusion, boundary_call_t boundary, 
                      split_call_t split, long seed, # rng_call_t rng,
                      double timestep, const double *t_observe, int t_observe_count, int *split_count,
                      double **split_times, double **split_points, double **split_weights, double *this_weights, double initial_weight, string scheme_name) nogil
