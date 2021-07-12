# distutils: language = c++

from pybatch.batch cimport PseudoParticleBatch
from pybatch.pybatch cimport PyPseudoParticleBatch

cdef class PyBatchKruells1(PyPseudoParticleBatch):
    def __cinit__(self, double x0, double y0, int N, double Tmax, double L, double Xsh, double a, double b):#, double x_min, double x_max):
        self._batch = <PseudoParticleBatch *>(new BatchKruells1(x0, y0, N, Tmax, L, Xsh, a, b))

    @staticmethod
    cdef BatchKruells1 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells1 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells1 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells1 *>self._batch;
            del tmp
            self._batch = NULL

