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

cdef class PyBatchKruellsB2(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruellsB2(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruellsB2 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruellsB2 *>_ptr

    def __dealloc__(self):
        cdef BatchKruellsB2 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruellsB2 *>self._batch;
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

cdef class PyBatchKruells10(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells10(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells10 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells10 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells10 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells10 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells11(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells11(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells11 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells11 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells11 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells11 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells12(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells12(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells12 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells12 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells12 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells12 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells13(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells13(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells13 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells13 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells13 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells13 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells14(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells14(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells14 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells14 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells14 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells14 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells15(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells15(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells15 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells15 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells15 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells15 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchKruells16(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchKruells16(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchKruells16 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchKruells16 *>_ptr

    def __dealloc__(self):
        cdef BatchKruells16 *tmp
        if not self._batch is NULL:
            tmp = <BatchKruells16 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchAchterberg1(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchAchterberg1(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchAchterberg1 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchAchterberg1 *>_ptr

    def __dealloc__(self):
        cdef BatchAchterberg1 *tmp
        if not self._batch is NULL:
            tmp = <BatchAchterberg1 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchAchterberg1KPPC(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchAchterberg1KPPC(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchAchterberg1KPPC *_cast_(PseudoParticleBatch *_ptr):
        return <BatchAchterberg1KPPC *>_ptr

    def __dealloc__(self):
        cdef BatchAchterberg1KPPC *tmp
        if not self._batch is NULL:
            tmp = <BatchAchterberg1KPPC *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchAchterberg2(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchAchterberg2(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchAchterberg2 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchAchterberg2 *>_ptr

    def __dealloc__(self):
        cdef BatchAchterberg2 *tmp
        if not self._batch is NULL:
            tmp = <BatchAchterberg2 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchAchterberg2KPPC(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchAchterberg2KPPC(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchAchterberg2KPPC *_cast_(PseudoParticleBatch *_ptr):
        return <BatchAchterberg2KPPC *>_ptr

    def __dealloc__(self):
        cdef BatchAchterberg2KPPC *tmp
        if not self._batch is NULL:
            tmp = <BatchAchterberg2KPPC *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchAchterberg2Implicit(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchAchterberg2Implicit(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchAchterberg2Implicit *_cast_(PseudoParticleBatch *_ptr):
        return <BatchAchterberg2Implicit *>_ptr

    def __dealloc__(self):
        cdef BatchAchterberg2Implicit *tmp
        if not self._batch is NULL:
            tmp = <BatchAchterberg2Implicit *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchAchterberg2SecondOrder(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchAchterberg2SecondOrder(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchAchterberg2SecondOrder *_cast_(PseudoParticleBatch *_ptr):
        return <BatchAchterberg2SecondOrder *>_ptr

    def __dealloc__(self):
        cdef BatchAchterberg2SecondOrder *tmp
        if not self._batch is NULL:
            tmp = <BatchAchterberg2SecondOrder *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchAchterberg2SecondOrder2(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchAchterberg2SecondOrder2(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchAchterberg2SecondOrder2 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchAchterberg2SecondOrder2 *>_ptr

    def __dealloc__(self):
        cdef BatchAchterberg2SecondOrder2 *tmp
        if not self._batch is NULL:
            tmp = <BatchAchterberg2SecondOrder2 *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchAchterberg2SemiImplicit(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchAchterberg2SemiImplicit(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchAchterberg2SemiImplicit *_cast_(PseudoParticleBatch *_ptr):
        return <BatchAchterberg2SemiImplicit *>_ptr

    def __dealloc__(self):
        cdef BatchAchterberg2SemiImplicit *tmp
        if not self._batch is NULL:
            tmp = <BatchAchterberg2SemiImplicit *>self._batch;
            del tmp
            self._batch = NULL

cdef class PyBatchAchterberg2SemiImplicit2(PyPseudoParticleBatch):
    def __cinit__(self, dict params):
        self._batch = <PseudoParticleBatch *>(new BatchAchterberg2SemiImplicit2(dict_to_map_string_double(params)))

    @staticmethod
    cdef BatchAchterberg2SemiImplicit2 *_cast_(PseudoParticleBatch *_ptr):
        return <BatchAchterberg2SemiImplicit2 *>_ptr

    def __dealloc__(self):
        cdef BatchAchterberg2SemiImplicit2 *tmp
        if not self._batch is NULL:
            tmp = <BatchAchterberg2SemiImplicit2 *>self._batch;
            del tmp
            self._batch = NULL
