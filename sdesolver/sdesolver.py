import copy
import time
import logging

import numpy as np

from sdesolver.sdecallback import SDECallbackBase, SDECallbackCoeff, SDECallbackBoundary
from sdesolver.loop.pyloop import py_integration_loop
from sdesolver.util.datastructures import SDEPPStateOldstyle

class SDE:
    """
    This class is intended to contain the physical (i.e. non-numerical)
    aspects of an SDE. Those are:
    - the coefficients (drift, diffusion)
    - parameters for those
    - boundaries
    - initial condition (list of 2-tuples of t and [x])
    - number of dimensions of the problem

    The coefficient and boundary callbacks can be set either by passing 
    callbacks to the constructor or by overriding the respective 
    functions when inheriting this class (or a combination of both).
    Both are automatically compiled to cfuncs if not done manually. This is
    done by the :py:class:`SDECallbackBoundary` and 
    :py:class:`SDECallbackCoeff` classes respectively

    :param ndim: Number of dimensions
    :type ndim: int

    :param initial_condition: A set of pseudo particles that should be propagated.
    :type initial_condition: List of tuples (t, [x])

    :param drift: Drift coefficient callback. If None, use the member function 
        with the same name. `(Default: None)`
    :type drift: function (out, t, x) -> None

    :param diffusion: Diffusion coefficient callback. If None, use the member function 
        with the same name. `(Default: None)`
    :type diffusion: function (out, t, x) -> None

    :param boundary: Escape boundary callback. If None, use the member function 
        with the same name. `(Default: None)`
    :type boundary: function (t, x) -> int


    """

    def __init__(self, ndim, initial_condition, drift=None, diffusion=None, boundary=None):
        self._set_callback(drift, "drift", SDECallbackCoeff)
        self._set_callback(diffusion, "diffusion", SDECallbackCoeff)
        self._set_callback(boundary, "boundary", SDECallbackBoundary)
        
        self.initial_condition = initial_condition
        self.ndim = ndim

    def _set_callback(self, arg, name, decorator):
        # decide where the callback is sourced from
        if not arg is None:
            cback = arg
        elif type(getattr(self, name)) is getattr(SDE, name):
            raise NotImplementedError(
                    "Either {} must be given on initialisation or it must be overriden by inheritance".format(name))
        else:
            cback = getattr(self, name)

        # check if it is numba compiled, if not try to do it now
        if not isinstance(cback, SDECallbackBase):
            logging.info("Callback for {} is not a numba-compiled C function. Trying to compile with default types.")
            cback = decorator(cback)

        # set the resulting callback
        setattr(self, name, cback)

    def set_parameters(self, parameters):
        """
        set all parameters with the same dict. Superflous parameters are
        ignored by SDECallbackBase.
        """
        self.drift.parameters = parameters
        self.diffusion.parameters = parameters
        self.boundary.parameters = parameters
    
    def drift(self, out, t, x):
        """
        The drift term of the SDE model.
        It is passed a time and a position (an array [x0, x1,...])
        and must write a drift vector with the same spatial
        dimensionality to out
        """
        pass

    def diffusion(self, out, t, x):
        """
        The diffusion term of the SDE model.
        It is passed a time and a position (an array [x0, x1,...])
        and must write a diffusion matrix with the same spatial
        dimensionality to out
        """
        pass

    def boundary(self, t, x):
        """
        function returning an int, depending on wether
        the particle hit a boundary. Must return 0 if no boundary is
        reached and some other value (apart from -1, which is reserved)
        if a boundary is reached. 
        """
        pass



class SDESolution:
    """
    Numerical solution of a stochastic differential equation.

    :param sde: The SDE that has been solved.
    :type sde: :py:class:`SDE`

    """
    def __init__(self, sde):
        self.sde = copy.copy(sde)
        self.observations = {}
        self._escaped_lists = {}
        #self._escaped_arrays = {}
        #self._escaped_updated = True

    def _add_observations(self, observation_times, positions, observation_count):
        for t, x in list(zip(observation_times, positions.copy()))[:observation_count]:
            if not t in self.observations:
                self.observations[t] = [x]
            else:
                self.observations[t].append(x)

    def __getitem__(self, s):
        """
        Get the observed pseudo particles at one of the times.
        """
        self.observations[s] = np.array(self.observations[s])
        return self.observations[s]

    def _add_escaped(self, t, x, boundary_state):
        if not boundary_state in self._escaped_lists:
            self._escaped_lists[boundary_state] = {'t' : [], 'x' : []}

        self._escaped_lists[boundary_state]['t'].append(t)
        self._escaped_lists[boundary_state]['x'].append(x)

        #self._escaped_updated = True
         
    @property
    def escaped(self):
        """
        A dict of lists of all escaped particles, for every return value
        of `sde.boundary`. The return value (boundary type) is the key of the
        dict.
        """
        #for k in self._escaped_lists.keys():
        #    self._escaped_arrays[k]['t'] = np.array(self._escaped[k]['t'])
        #    self._escaped_arrays[k]['x'] = np.array(self._escaped[k]['x'])

        #self._escaped_updated = False
        #return self._escaped_arrays
        return self._escaped_lists

    @property
    def observation_times(self):
        """
        A list of the times at which the particle distribution has been observed
        """
        return self.observations.keys()

    @property
    def boundary_states(self):
        """
        A list of all boundary types (return values of the `sde.boundary` call)
        that have been returned.
        """
        return self._escaped_lists.keys()


    def get_oldstyle_pps(self, observation_time):
        """
        Get a list of :py:class:`SDEPPStateOldstyle` for use with the old
        processing code.

        :param observation_time: The time at which the particles states should be returned
        :type observation_time: float
        """
        pstates = []
        for pp in self[observation_time]:
            pstates.append(SDEPPStateOldstyle(observation_time, np.array([pp]).T, 0))

        return pstates



class SDESolver:
    """
    Solver of a stochastic differential equation.

    :param scheme: Numerical scheme to use. See :cpp:func:`scheme_registry_lookup` for available schemes. 
    :type scheme: string

    :param noise_term: Currently not in use. Default: None

    """
    def __init__(self, scheme, noise_term=None):
        self.scheme = scheme
        self.noise_term = noise_term
        if not noise_term is None:
            print("WARNING: noise_term is currently ignored")

    def solve(self, sde, timestep, observation_times):
        res = SDESolution(sde)

        observation_times = np.array(sorted(observation_times))
        seeds = list(range(len(sde.initial_condition)))
        """
        Solve a SDE with this solver using a linear single-thread approach.

        :param sde: The SDE to be solved.
        :type sde: :py:class:`SDE`

        :param timestep: Integration timestep
        :type timestep: float

        :param observation_times: Times at which the pseudo particle distribution will be observed, i.e. stored.
        :type observation_times: list of float
        """

        start = time.perf_counter()
        time_cpp = 0

        observations_contiguous = np.empty(len(observation_times) * sde.ndim)
        # these arrays are needed because we want to give the loop pointers
        t_array = np.empty(1, dtype=np.float64)
        observation_count_array = np.empty(1, dtype=np.int32)

        init_copy = copy.copy(sde.initial_condition)

        for (pp_t, pp_x), seed in zip(init_copy, seeds):
            start_cpp = time.perf_counter()
            t_array[0] = pp_t

            boundary_state = py_integration_loop(observations_contiguous, observation_count_array, t_array, pp_x,
                                     sde.drift.address, sde.diffusion.address, sde.boundary.address,
                                     seed, timestep, 
                                     observation_times, self.scheme)

            end_cpp = time.perf_counter()
            time_cpp += end_cpp - start_cpp

            if boundary_state != 0:
                res._add_escaped(t_array[0], pp_x, boundary_state)

            res._add_observations(observation_times, observations_contiguous.reshape((-1, sde.ndim)), observation_count_array[0])

        end = time.perf_counter()
        logging.info("Runtime for pseudoparticle propagation: {}us".format(time_cpp * 1e6))
        logging.info("Runtime for propagation and transformation: {}us".format((end - start) * 1e6))

        return res
