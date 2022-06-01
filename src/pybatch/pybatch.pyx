# distutils: language = c++

from pybatch.batch cimport PseudoParticleBatch

from pybatch.pseudoparticlestate cimport PseudoParticleState
from pybatch.pypseudoparticlestate cimport PyPseudoParticleState

from eigency.core cimport *
from libcpp.vector cimport vector
from libcpp.map cimport map as cpp_map
from libcpp.string cimport string

import numpy as np

cimport numpy as np
from cysignals.signals cimport sig_on, sig_off
#https://github.com/cython/cython/wiki/WrappingSetOfCppClasses

cdef cpp_map[string, double] dict_to_map_string_double(dict d):
    cdef cpp_map[string, double] m
    for k_, v_ in d.items():
        m[k_.encode('utf-8')] = v_
    return m

cdef class PyPseudoParticleBatch:
    """
    Base class wrapping a :cpp:class:`PseudoParticleBatch`

    Inherit the cython wrappers for specialized :cpp:class:`PseudoParticleBatch` C++ classes from this class.
    The inheriting scheme resembles the inheritance on the C++ side.

    Inherited classes need to define __cinit__ to initialize the correctly typed PseudoParticleBatch
    instance that is stored in _batch. __init__ in this class only defaults some attributes.

    :param dict params: Batch parameters
    """
    def __init__(self, params):
        """
        Initialize the Node

        :param dict params: The params given to the C++ backend
        """
        self._states = None
        self._integrator_values = None
        self._reconstructed = False
        self._params = params

    def __dealloc__(self):
        if not self._batch is NULL:
            del self._batch
            self._batch = NULL

    def run(self, int particle_count=-1, int nthreads=1):
        """
        Run all or a given amount of pseudo particles in the batch.

        This method is not available for reconstructed (unpickled) instances.

        :param int particle_count: *(optional)* The number of pseudo particles to run. If -1, all particles are run.
        :return: *(int)* number of finished particles
        """
        self._raise_reconstructed_exception()

        sig_on()
        r = (<PseudoParticleBatch *>(self._batch)).run(particle_count, nthreads)
        sig_off()

        return r

    def step_all(self, int steps=1):
        """
        Simulate a given number of steps for every pseudo particle in the batch.

        This method is not available for reconstructed (unpickled) instances.

        :param int steps: Number of steps to simulate
        :return: *(int)* Number of finished particles
        """
        self._raise_reconstructed_exception()

        sig_on()
        r = (<PseudoParticleBatch *>(self._batch)).step_all(steps)
        sig_off()

        return r

    @property
    def unfinished_count(self):
        """
        Number of unfished pseudo particles

        This method is not available for reconstructed (unpickled) instances.

        :return: *(int)* Number of unfinished particles
        """
        self._raise_reconstructed_exception()
        return (<PseudoParticleBatch *>(self._batch)).unfinished_count()
    
    def _raise_reconstructed_exception(self):
        if self._reconstructed is True:
            raise ValueError("Cannot access C++ level functions on reconstructed objects")

    def state(self, int index):
        """
        Get a state by index.

        :return: :py:class:`pybatch.pypseudoparticlestate.PyPseudoParticleState` 
        """
        return self._states[index]

    @property
    def states(self):
        """
        All pseudo particle states

        :return: [:py:class:`pybatch.pypseudoparticlestate.PyPseudoParticleState`]
        """
        if self._states is None:
            self._fetch_states()
        return self._states

    @property
    def integrator_values(self):
        """
        Values of the integrators.

        :return: List of N lists of each M `double` values. N is the number of pseudo particles, M is the number of integrators.
        """
        if self._integrator_values is None:
            self._fetch_integrator_values()
        return self._integrator_values

    def _fetch_integrator_values(self):
        self._integrator_values = []
        cdef vector[vector[double]] vvlist = (<PseudoParticleBatch *>(self._batch)).get_integrator_values()

        for vlist in vvlist:
            rlist = []
            for v in vlist:
                rlist.append(v)
            self._integrator_values.append(rlist)

    def _fetch_states(self):
        cdef vector[PseudoParticleState] vlist 
        self._states = []
        vlist = (<PseudoParticleBatch *>(self._batch)).states()
        for s in vlist:
            self._states.append(PyPseudoParticleState.from_cobj(s))

    def _set_vars(self, states, integrator_values):
        self._reconstructed = True
        self._states = states
        self._integrator_values = integrator_values


    # pickling doesnt create a new batch with full functionality
    # it only restores the states and integrator values
    def __reduce__(self):
        if type(self) == PyPseudoParticleBatch:
            raise ValueError("Cannot __reduce__ a PyPseudoParticleBatch (you must inherit from this class)")

        print("Reducing...")
        return type(self)._reconstruct, (self._params, self.states, self.integrator_values)

    @classmethod
    def _reconstruct(cls, params, states, integrator_values):
        instance = cls(params)
        instance._set_vars(states, integrator_values)
        return instance
