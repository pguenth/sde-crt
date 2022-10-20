# cython: profile=True
# distutils: language = c++
# distutils: sources = src/cscheme.cpp


import numpy as np
cimport numpy as np
np.import_array()

from eigency.core cimport *
from cython.operator cimport dereference

from cscheme cimport SpaceTimePoint



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

cdef class PySpaceTimePoint:
    def __cinit__(self, double t, np.ndarray[np.float64_t] x):
        self._spacetimepoint = new SpaceTimePoint(t, Map[VectorXd](x))
        self._owned = True

    @staticmethod
    cdef object _from_cptr(SpaceTimePoint *ptr, bool owned):
        cdef PySpaceTimePoint stp = PySpaceTimePoint.__new__(PySpaceTimePoint)
        stp._spacetimepoint = ptr
        stp._owned = owned
        return stp

    def __dealloc__(self):
        if self._owned:
            del self._spacetimepoint

    def __reduce__(self):
        return type(self)._reconstruct, (self.t, self.x)
    
    @classmethod
    def _reconstruct(cls, t, x):
        return cls(t, x)

    @property
    def t(self):
        return self._spacetimepoint.t

    @property
    def x(self):
        return ndarray(self._spacetimepoint.x).T[0]

cpdef Point sde_scheme_euler_cpp(double t, np.ndarray[np.float64_t] x, np.ndarray[np.float64_t] rndvec, double timestep, long drift_addr, long diffusion_addr):
    p = PySpaceTimePoint(t, x)
    stp = scheme_euler(dereference(p._spacetimepoint), Map[VectorXd](rndvec), timestep, <coeff_call_t>(<void *>drift_addr), <coeff_call_t>(<void *>diffusion_addr))
    return Point(stp.t, ndarray(stp.x).T[0])



cdef Point sde_scheme_euler_c(double t, double[:] x, double[:] rndvec, double timestep, callback_t drift, callback_t diffusion):
    #t, x = pp.t, pp.x
    cdef double t_new = t + timestep
    #drift_term = timestep * drift(t, x)
    #diff_term = np.dot(diffusion(t, x), rndvec)
    
    cdef Py_ssize_t ndim = x.shape[0]

    cdef np.ndarray[np.float64_t] drift_np = np.zeros(ndim, dtype=np.double)
    cdef np.double_t *drift_ptr = <np.double_t *>drift_np.data
    drift(drift_ptr, t, &x[0])

    cdef np.ndarray[np.float64_t] diff_np = np.zeros(ndim**2, dtype=np.double)
    cdef np.double_t *diff_ptr = <np.double_t *>diff_np.data
    diffusion(diff_ptr, t, &x[0])

    x_new = x + timestep * drift_np + np.dot(np.reshape(diff_np, (ndim, ndim)), rndvec) * np.sqrt(timestep)
    return Point(t_new, x_new)

cpdef Point sde_scheme_euler_cython(double t, double[:] x, double[:] rndvec, double timestep, long drift_addr, long diffusion_addr):
    return sde_scheme_euler_c(t, x, rndvec, timestep, <callback_t>(<void *>drift_addr), <callback_t>(<void *>diffusion_addr))
