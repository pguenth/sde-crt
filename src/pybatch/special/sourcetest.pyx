# distutils: language = c++

from pybatch.batch cimport PseudoParticleBatch
from pybatch.pybatch cimport PyPseudoParticleBatch

cdef class PyBatchSourcetest(PyPseudoParticleBatch):
    def __cinit__(self, double x0, int N, double Tmax, double x_min, double x_max):
        self._batch = <PseudoParticleBatch *>(new BatchSourcetest(x0, N, Tmax, x_min, x_max))

    @staticmethod
    cdef BatchSourcetest *_cast_(PseudoParticleBatch *_ptr):
        return <BatchSourcetest *>_ptr

    def __dealloc__(self):
        cdef BatchSourcetest *tmp
        if not self._batch is NULL:
            tmp = <BatchSourcetest *>self._batch;
            del tmp
            self._batch = NULL

    def integrate(self):
        return (<BatchSourcetest *>self._batch).integrate()
