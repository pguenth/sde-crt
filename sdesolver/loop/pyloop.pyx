# cython: profile=True
# cython: embedsignature = True
# distutils: language = c++
# distutils: sources = sdesolver/loop/loop.cpp sdesolver/loop/scheme.cpp sdesolver/loop/broyden.cpp


import numpy as np
cimport numpy as np
np.import_array()

from eigency.core cimport *
from cython.operator cimport dereference
cimport cython

from sdesolver.loop.loop cimport integration_loop_p, boundary_call_t, rng_call_t, coeff_call_t, split_call_t

from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
cpdef int py_integration_loop(double[:] x_obs, int[:] observation_count, double[:] t, np.ndarray[np.float64_t] x, long drift_addr,
                              long diffusion_addr, long boundary_addr, long split_addr, long seed,#long rng_addr, 
                              double timestep, double[:] t_obs, list split_times, list split_points, list split_weights, double[:] this_weights, double[:] weight, string scheme_name):
    """
    split_points should be an empty list (which gets filled by this function)
    """
    cdef Map[VectorXd] x_map = Map[VectorXd](x)
    cdef double **split_points_p = <double **>malloc(sizeof(double *)) # flattened split_points array is stored here
    cdef double **split_times_p = <double **>malloc(sizeof(double *))
    cdef double **split_weights_p = <double **>malloc(sizeof(double *))
    cdef int split_count
    cdef int i
    cdef int ndim = x.size

    with nogil:
        boundary_state = integration_loop_p(&x_obs[0], &observation_count[0], &t[0], x_map,
            <coeff_call_t>(<void *>drift_addr), <coeff_call_t>(<void *>diffusion_addr),
            <boundary_call_t>(<void *>boundary_addr), <split_call_t>(<void *>split_addr), 
            seed,# <rng_call_t>(<void *>rng_addr),
            timestep, &t_obs[0], len(t_obs), &split_count, split_times_p, split_points_p, split_weights_p, &this_weights[0], &weight[0], scheme_name)

    for i in range(split_count):
        p = []
        for j in range(ndim):
            p.append(dereference(split_points_p)[i * ndim + j])
        split_times.append(dereference(split_times_p)[i])
        split_weights.append(dereference(split_weights_p)[i])
        split_points.append(p)

    free(dereference(split_points_p))
    free(dereference(split_times_p))
    free(dereference(split_weights_p))
    free(split_points_p)
    free(split_times_p)
    free(split_weights_p)

    return boundary_state
