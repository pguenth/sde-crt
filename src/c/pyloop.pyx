# cython: profile=True
# distutils: language = c++
# distutils: sources = src/c/loop.cpp src/c/scheme.cpp


import numpy as np
cimport numpy as np
np.import_array()

from eigency.core cimport *
from cython.operator cimport dereference

from loop cimport integration_loop_p, boundary_call_t, rng_call_t, coeff_call_t

from libcpp.string cimport string
from libcpp.vector cimport vector

cpdef int py_integration_loop(double[:] x_obs, int[:] observation_count, double[:] t, np.ndarray[np.float64_t] x, long drift_addr,
                   long diffusion_addr, long boundary_addr, long seed,#long rng_addr, 
                   double timestep, double[:] t_obs, string scheme_name):

    return integration_loop_p(&x_obs[0], &observation_count[0], &t[0], Map[VectorXd](x),
        <coeff_call_t>(<void *>drift_addr), <coeff_call_t>(<void *>diffusion_addr),
        <boundary_call_t>(<void *>boundary_addr), seed,# <rng_call_t>(<void *>rng_addr),
        timestep, &t_obs[0], len(t_obs), scheme_name)
