# distutils: language = c++

from pybatch.batch cimport PseudoParticleBatch
from pybatch.pybatch cimport PyPseudoParticleBatch
from pybatch.pybatch cimport dict_to_map_string_double

cdef class PyBatchKruells921(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells921(dict_to_map_string_double(params)))

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
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells922(dict_to_map_string_double(params)))

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
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells923(dict_to_map_string_double(params)))

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
    def __cinit__(self, dict params):
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

