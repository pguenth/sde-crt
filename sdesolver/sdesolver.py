import copy
import time
import logging
import sys
import pprint

from multiprocessing.pool import ThreadPool
from multiprocessing import Pipe, connection
from threading import Thread
import threading

import numpy as np

from sdesolver.sdecallback import SDECallbackBase, SDECallbackCoeff, SDECallbackBoundary, SDECallbackSplit
from sdesolver.loop.pyloop import py_integration_loop
from sdesolver.util.datastructures import SDEPPStateOldstyle
from sdesolver.supervisor import Supervisor

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
    :py:class:`SDECallbackCoeff` classes respectively. FIXME: Is this tested?
    It will probably not work because of the self parameter in the inherited
    functions.

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

    def __init__(self, ndim, initial_condition, drift=None, diffusion=None, boundary=None, split=None):
        self._set_callback(drift, "drift", SDECallbackCoeff)
        self._set_callback(diffusion, "diffusion", SDECallbackCoeff)
        self._set_callback(boundary, "boundary", SDECallbackBoundary)
        self._set_callback(split, "split", SDECallbackSplit)
        
        self.initial_condition = initial_condition
        self.ndim = ndim

    def _set_callback(self, arg, name, decorator):
        # decide where the callback is sourced from
        if not arg is None:
            cback = arg
        elif name not in ['split', 'boundary'] and getattr(type(self), name) is getattr(SDE, name):
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
        self.split.parameters = parameters
    
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
        return 0

    def split(self, t, x, last_t, last_x):
        """
        function returning a bool, depending on wether
        the particle should be splitted at the current point. This callback
        is also passed the time and position of the last splitting event
        for this particle.
        """
        return False

    def __eq__(self, v):
        if len(v.initial_condition) != len(self.initial_condition):
            print("lengths not equal")
            return False

        for (d0, a0), (d1, a1) in zip(v.initial_condition, self.initial_condition):
            if d0 != d1:
                return False
            if not np.array_equal(a0, a1):
                return False

        if v.ndim != self.ndim:
            print("ndim ineq")
            return False

        if self.drift != v.drift:
            print("drift ineq")
            return False
        if self.diffusion != v.diffusion:
            print("diffusion ineq")
            return False
        if self.boundary != v.boundary:
            print("boundary ineq")
            return False
        if self.split != v.split:
            print("split ineq")
            return False

        return True

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

    def _add_observations(self, observation_times, positions, weights):
        for t, x, w in list(zip(observation_times, positions.copy(), weights)):
            if not t in self.observations:
                self.observations[t] = {'x': [x], 'weights': [w]}
            else:
                self.observations[t]['x'].append(x)
                self.observations[t]['weights'].append(w)

    def __getitem__(self, s):
        """
        Get the observed pseudo particles at one of the times.
        """
        self.observations[s] = {k : np.array(v) for k, v in self.observations[s].items()}
        return self.observations[s]

    def _add_escaped(self, t, x, weight, boundary_state):
        if not boundary_state in self._escaped_lists:
            self._escaped_lists[boundary_state] = {'t' : [], 'x' : [], 'weights': []}

        self._escaped_lists[boundary_state]['t'].append(t)
        self._escaped_lists[boundary_state]['x'].append(x)
        self._escaped_lists[boundary_state]['weights'].append(weight)

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

    @staticmethod
    def solve_one(pp_t, pp_x, seed, observation_times, sde, timestep, scheme, initial_weight=1.0, supervisor_pipe=None):
        observations_contiguous = np.empty(len(observation_times) * sde.ndim)
        # these arrays are needed because we want to give the loop pointers
        t_array = np.empty(1, dtype=np.float64)
        observation_count_array = np.empty(1, dtype=np.int32)

        start_cpp = time.perf_counter()
        t_array[0] = pp_t
        split_times = []
        split_points = []

        boundary_state = py_integration_loop(observations_contiguous, observation_count_array, t_array, pp_x,
                                 sde.drift.address, sde.diffusion.address, sde.boundary.address, sde.split.address,
                                 seed, timestep, observation_times, split_times, split_points, scheme)

        end_cpp = time.perf_counter()
        time_cpp = end_cpp - start_cpp

        if not supervisor_pipe is None:
            # using this reduces performance significantly (x2-x4)
            supervisor_pipe.send({'thread_id': threading.get_ident(), 'time_cpp': time_cpp, 'time_phys': t_array[0] - pp_t, 'split': len(split_times), 'particles' : 0})

        observations = observations_contiguous.reshape((-1, sde.ndim))[:observation_count_array[0]]

        weight = initial_weight
        weights = []
        observation_time_it = iter(observation_times)
        observation_time = next(observation_time_it)

        for split_t in split_times:
            # store weight if an observation was made between the last and the current split
            while split_t > observation_time:
                weights.append(weight)
                try:
                    observation_time = next(observation_time_it)
                except StopIteration:
                    break

            # decrease weight
            weight /= 2

            #if observation_index >= len(observation_times):
            #    # catch round-off errors (split on the last timestep can lead
            #    # to the split time being larger than the observation time
            #    print(f"no observations for this pseudoparticle {split_t}, {split_times}, {split_points}")
            #    break

        while len(weights) < len(observation_times):
            weights.append(weight)
            
        split_weights = initial_weight / 2**(np.arange(len(split_times)) + 1)


        particle_info = {'final_t': t_array[0],
                         'final_x': pp_x,
                         'observations': observations,
                         'boundary_state': boundary_state,
                         'weights': weights
                        }

        split_info = list(zip(split_times, np.array(split_points), split_weights)) #{'t': split_times, 'x': split_points, 'w': split_weights}
        if not len(split_times) == 0:
            if np.min(split_times) <= pp_t:
                print(f"Particle back-splitted")
            if np.max(split_times) > np.max(observation_times):
                print(f"particle behind last observation time")
        #print("sw, w",particle_info, split_info)

        return particle_info, split_info, time_cpp

    @staticmethod
    def solve_slice(sde, timestep, observation_times, scheme, seeds_slice, init, supervisor_pipe=None):
        particles = []
        splits = []

        last_p = 0
        last_s = 0
        total_time_cpp = 0

        for (pp_t, pp_x, w), seed in zip(init, seeds_slice):
            # slice observation_times to avoid multiple observations of particles at their start time
            observation_times_slice = np.array([t for t in observation_times if t >= pp_t])
            pinfo, sinfo, time_cpp = SDESolver.solve_one(pp_t, pp_x, seed, observation_times_slice, sde, timestep, scheme, w, )#supervisor_pipe)
            pinfo['observation_times_slice'] = observation_times_slice
            particles.append(pinfo)
            splits += sinfo
            total_time_cpp += time_cpp

            if supervisor_pipe is not None and len(particles) % 100 == 0:
                supervisor_pipe.send({'type': 'slice', 'thread_id': threading.get_ident(), 'time_cpp': total_time_cpp, 'time_phys' : 0, 'particles': len(particles) - last_p, 'splits': len(splits) - last_s})
                last_p = len(particles)
                last_s = len(splits)
                total_time_cpp = 0

        if supervisor_pipe is not None:
            supervisor_pipe.send({'type': 'slice', 'thread_id': threading.get_ident(), 'time_cpp': total_time_cpp, 'time_phys' : 0, 'particles': len(particles) - last_p, 'splits': len(splits) - last_s})
            supervisor_pipe.close()
        return particles, splits

    def schedule_slices(self, sde, timestep, observation_times, seed_stream, init, pool, sderesult, supervisor=None):
        """
        init: list of 3-tuples (t, [x0, x1, ...], weight)
        """
        nthreads = pool._processes
        asyncresults = []

        # since this function's call order (from splits) depends on the runtime of the worker threads,
        # the seeding procedure used will probably lead to non-deterministic behaviour even for fixed seeds
        # this could be fixable by "inheriting" some kind of state from here to the recursive calls

        if len(init) == 0:
            return

        collected_split_starts = []

        for mod in range(nthreads):
            if supervisor is not None:
                recv, send = Pipe(duplex=False)
                supervisor.attach(recv)
            else:
                recv, send = None, None

            init_slice = init[mod::nthreads]
            seeds = seed_stream.integers(sys.maxsize, size=len(init_slice))
            slice_args = (sde, timestep, observation_times, self.scheme,
                          seeds, init_slice, send)

            asr = pool.apply_async(self.solve_slice, slice_args)
            asyncresults.append(asr)

        for asr in asyncresults:
            particles, splits = asr.get()
            split_t = np.array([t for t, _, _ in splits])

            for par in particles:
                if par['boundary_state'] != 0:
                    # is picking the last weight right?
                    sderesult._add_escaped(par['final_t'], par['final_x'], par['weights'][-1], par['boundary_state']) 

                sderesult._add_observations(par['observation_times_slice'], par['observations'], par['weights'])

            collected_split_starts += splits

            self.schedule_slices(sde, timestep, observation_times, seed_stream, splits, pool, sderesult, supervisor)

        import pickle
        with open("splits.pickle", mode="wb") as f:
            pickle.dump(collected_split_starts, f)

    def _format_time(self, t):
        return "{}s{}ms{}us".format(int(t), int(int(t * 1e3) % 1e3), int(int(t * 1e6) % 1e6))

    def solve(self, sde, timestep, observation_times, seed_stream=None, nthreads=4, supervise=True):
        """
        Solve a SDE with this solver using multiple threads.

        :param sde: The SDE to be solved.
        :type sde: :py:class:`SDE`

        :param timestep: Integration timestep
        :type timestep: float

        :param observation_times: Times at which the pseudo particle distribution will be observed, i.e. stored.
        :type observation_times: list of float

        :param seeds: Seeds for every sde solution. If None, use 0, 1, 2, ...
        :type seeds: None or list of int with length of observation_times

        :param nthreads: Number of threads. If -1, use one thread per solution.
        :type nthreads: int
        """
        sdesolution = SDESolution(sde)

        observation_times = np.array(sorted(observation_times))

        if seed_stream is None:
            from numpy.random import Generator, PCG64
            seed_stream = Generator(PCG64(1234567890))

        start = time.perf_counter()

        init_copy = []
        for pp_t, pp_x in sde.initial_condition:
            init_copy.append((pp_t, copy.copy(pp_x), 1.0))

        if nthreads == -1:
            pool = ThreadPool()
        else:
            pool = ThreadPool(processes=nthreads)

        if supervise:
            supervisor = Supervisor()
            supervisor_thread = Thread(target=supervisor.loop)
            supervisor_thread.start()
        else:
            supervisor = None

        self.schedule_slices(sde, timestep, observation_times, seed_stream, init_copy, pool, sdesolution, supervisor=supervisor)

        end = time.perf_counter()

        if supervise:
            supervisor.interrupt()
            supervisor_thread.join()
            logging.info("Supervisor thread joined")

        total_time = end - start
        time_cpp = supervisor.data[1]['time_cpp']
        logging.info("Total simulation runtime: {}".format(self._format_time(total_time)))
        logging.info("Total cpp runtime: {}".format(self._format_time(time_cpp)))
        logging.info("python overhead (= 1 - cpp/(nthreads * total)): {:.3g}%".format(100 * (1 - time_cpp / (nthreads * total_time))))


        return sdesolution

