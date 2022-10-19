# cython: profile=True
# distutils: language = c++


import numpy as np
cimport numpy as np
np.import_array()

from eigency.core cimport *



#DTYPE = np.double
#ctypedef np.double_t DTYPE_t
#ctypedef np.ndarray (*callback_t)(double t, np.ndarray x)
#ctypedef void (*callback_t)(np.ndarray out, double t, np.ndarray x)
ctypedef void (*callback_t)(double *out, double t, double *x)

cdef class Point:
    cdef public double t 
    cdef public np.ndarray x

    def __init__(self, t, x):
        self.t = t
        self.x = x

cdef Point sde_scheme_euler_c(double t, double[:] x, double[:] rndvec, double timestep, callback_t drift, callback_t diffusion):
    #t, x = pp.t, pp.x
    cdef double t_new = t + timestep
    #drift_term = timestep * drift(t, x)
    #diff_term = np.dot(diffusion(t, x), rndvec)
    
    cdef Py_ssize_t ndim = x.shape[0]

    cdef np.ndarray[np.float64_t] drift_np = np.zeros(ndim, dtype=np.doublec)
    cdef np.double_t *drift_ptr = <np.double_t *>drift_np.data
    drift(drift_ptr, t, &x[0])

    cdef np.ndarray[np.float64_t] diff_np = np.zeros((ndim, ndim), dtype=np.doublec)
    cdef np.double_t *diff_ptr = <np.double_t *>diff_np.data
    diffusion(diff_ptr, t, &x[0])

    x_new = x + timestep * drift_np# + np.dot(diff_np, rndvec) * np.sqrt(timestep)
    return Point(t_new, x_new)

cpdef Point sde_scheme_euler_cython(double t, double[:] x, double[:] rndvec, double timestep, long drift_addr, long diffusion_addr):
    return sde_scheme_euler_c(t, x, rndvec, timestep, <callback_t>(<void *>drift_addr), <callback_t>(<void *>diffusion_addr))
