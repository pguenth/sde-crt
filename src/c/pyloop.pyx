# cython: profile=True
# distutils: language = c++
# distutils: sources = loop.cpp scheme.cpp


import numpy as np
cimport numpy as np
np.import_array()

from eigency.core cimport *
from cython.operator cimport dereference

from loop cimport ploop_pointer, boundary_call_t, rng_call_t, coeff_call_t

from libcpp.string cimport string
from libcpp.vector cimport vector



#DTYPE = np.double
#ctypedef np.double_t DTYPE_t
#ctypedef np.ndarray (*callback_t)(double t, np.ndarray x)
#ctypedef void (*callback_t)(np.ndarray out, double t, np.ndarray x)
#ctypedef void (*callback_t)(double *out, double t, double *x)

cpdef void pyploop(double [:] x_obs, double t0, np.ndarray[np.float64_t] x0, long drift_addr, long diffusion_addr, long boundary_addr, long rng_addr, double timestep, double[:] t_obs, string scheme_name):

    #cdef vector[VectorXd] x_obs_cpp
    #cdef vector[double] t_obs_cpp

    #for t in t_obs:
    #    t_obs_cpp.push_back(t)


    #ploop_pointer(x_obs_cpp, t0, Map[VectorXd](x0),
    #    <coeff_call_t>(<void *>drift_addr), <coeff_call_t>(<void *>diffusion_addr),
    #    <boundary_call_t>(<void *>boundary_addr), <rng_call_t>(<void *>rng_addr),
    #    timestep, t_obs_cpp, scheme_name)

    ploop_pointer(&x_obs[0], t0, Map[VectorXd](x0),
        <coeff_call_t>(<void *>drift_addr), <coeff_call_t>(<void *>diffusion_addr),
        <boundary_call_t>(<void *>boundary_addr), <rng_call_t>(<void *>rng_addr),
        timestep, &t_obs[0], len(t_obs), scheme_name)


    #cdef int ndim = len(x0)
    #cdef double[:, :] x_obs_ret = np.empty((x_obs_cpp.size(), ndim))

    #for i in range(x_obs_cpp.size()):
    #    for j in range(ndim):
    #        x_obs_ret[i, j] = x_obs_cpp[i](j)




#cdef Point sde_scheme_euler_c(double t, double[:] x, double[:] rndvec, double timestep, callback_t drift, callback_t diffusion):
#    #t, x = pp.t, pp.x
#    cdef double t_new = t + timestep
#    #drift_term = timestep * drift(t, x)
#    #diff_term = np.dot(diffusion(t, x), rndvec)
#    
#    cdef Py_ssize_t ndim = x.shape[0]
#
#    cdef np.ndarray[np.float64_t] drift_np = np.zeros(ndim, dtype=np.double)
#    cdef np.double_t *drift_ptr = <np.double_t *>drift_np.data
#    drift(drift_ptr, t, &x[0])
#
#    cdef np.ndarray[np.float64_t] diff_np = np.zeros(ndim**2, dtype=np.double)
#    cdef np.double_t *diff_ptr = <np.double_t *>diff_np.data
#    diffusion(diff_ptr, t, &x[0])
#
#    x_new = x + timestep * drift_np + np.dot(np.reshape(diff_np, (ndim, ndim)), rndvec) * np.sqrt(timestep)
#    return Point(t_new, x_new)
#
#cpdef Point sde_scheme_euler_cython(double t, double[:] x, double[:] rndvec, double timestep, long drift_addr, long diffusion_addr):
#    return sde_scheme_euler_c(t, x, rndvec, timestep, <callback_t>(<void *>drift_addr), <callback_t>(<void *>diffusion_addr))
