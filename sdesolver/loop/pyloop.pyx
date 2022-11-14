# cython: profile=True
# cython: embedsignature = True
# distutils: language = c++
# distutils: sources = sdesolver/loop/loop.cpp sdesolver/loop/scheme.cpp


import numpy as np
cimport numpy as np
np.import_array()

from eigency.core cimport *
from cython.operator cimport dereference
cimport cython

from sdesolver.loop.loop cimport integration_loop_p, boundary_call_t, rng_call_t, coeff_call_t

from libcpp.string cimport string
from libcpp.vector cimport vector

@cython.boundscheck(False)
cpdef int py_integration_loop(double[:] x_obs, int[:] observation_count, double[:] t, np.ndarray[np.float64_t] x, long drift_addr,
                   long diffusion_addr, long boundary_addr, long seed,#long rng_addr, 
                   double timestep, double[:] t_obs, string scheme_name):
    """
    asdf
    """
    cdef Map[VectorXd] x_map = Map[VectorXd](x)

    with nogil:
        boundary_state = integration_loop_p(&x_obs[0], &observation_count[0], &t[0], x_map,
            <coeff_call_t>(<void *>drift_addr), <coeff_call_t>(<void *>diffusion_addr),
            <boundary_call_t>(<void *>boundary_addr), seed,# <rng_call_t>(<void *>rng_addr),
            timestep, &t_obs[0], len(t_obs), scheme_name)

    return boundary_state
