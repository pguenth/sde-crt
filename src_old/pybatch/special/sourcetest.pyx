# distutils: language = c++

from pybatch.batch cimport PseudoParticleBatch
from pybatch.pybatch cimport PyPseudoParticleBatch
from pybatch.pybatch cimport dict_to_map_string_double

cdef class PyBatchSourcetest(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchSourcetest(dict_to_map_string_double(params)))

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
