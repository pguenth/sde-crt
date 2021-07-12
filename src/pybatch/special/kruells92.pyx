# distutils: language = c++

from pybatch.batch cimport PseudoParticleBatch
from pybatch.pybatch cimport PyPseudoParticleBatch
from pybatch.pybatch cimport dict_to_map_string_double

cdef class PyBatchKruells921(PyPseudoParticleBatch):
    def __cinit__(self, double x0, double y0, int N, double Tmax, double Tesc):#, double x_min, double x_max):
        self._batch = <PseudoParticleBatch *>(new BatchKruells921(x0, y0, N, Tmax, Tesc))

    @staticmethod
    cdef BatchKruells921 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells921 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells921 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells921 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells922(PyPseudoParticleBatch):
    def __cinit__(self, double x0, double y0, int N, double Tmax, double dxs, double Kpar, double r, double Vs, double dt, double beta_s):#, double x_min, double x_max):
        self._batch = <PseudoParticleBatch *>(new BatchKruells922(x0, y0, N, Tmax, dxs, Kpar, r, Vs, dt, beta_s))

    @staticmethod
    cdef BatchKruells922 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells922 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells922 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells922 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells923(PyPseudoParticleBatch):
    def __cinit__(self, double x0, double y0, double r_inj, double Tmax, double dxs, double Kpar, double r, double Vs, double dt, double beta_s):#, double x_min, double x_max):
        self._batch = <PseudoParticleBatch *>(new BatchKruells923(x0, y0, r_inj, Tmax, dxs, Kpar, r, Vs, dt, beta_s))

    @staticmethod
    cdef BatchKruells923 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells923 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells923 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells923 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells924(PyPseudoParticleBatch):
    def __cinit__(self, dict params):#, double x_min, double x_max):
        self._batch = <PseudoParticleBatch *>(new BatchKruells924(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells924 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells924 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells924 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells924 *>self._batch;
            del tmp
            self._batch = NULL

