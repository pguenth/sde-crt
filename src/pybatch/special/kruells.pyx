# distutils: language = c++

from pybatch.batch cimport PseudoParticleBatch
from pybatch.pybatch cimport PyPseudoParticleBatch
from pybatch.pybatch cimport dict_to_map_string_double

cdef class PyBatchKruells1(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells1(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells1 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells1 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells1 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells1 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells2(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells2(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells2 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells2 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells2 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells2 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells3(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells3(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells3 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells3 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells3 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells3 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells4(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells4(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells4 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells4 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells4 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells4 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells5(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells5(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells5 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells5 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells5 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells5 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells6(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells6(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells6 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells6 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells6 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells6 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells7(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells7(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells7 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells7 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells7 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells7 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells8(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells8(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells8 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells8 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells8 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells8 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruellsB1(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruellsB1(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruellsB1 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruellsB1 *>_ptr

    def __dealloc__(self):
        cdef BatchKruellsB1 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruellsB1 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruellsC1(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruellsC1(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruellsC1 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruellsC1 *>_ptr

    def __dealloc__(self):
        cdef BatchKruellsC1 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruellsC1 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells9(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells9(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells9 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells9 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells9 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells9 *>self._batch;
            del tmp
            self._batch = NULL

