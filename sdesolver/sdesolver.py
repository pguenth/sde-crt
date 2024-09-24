import copy
import time
import logging
import sys
import pprint
import random

from multiprocessing.pool import ThreadPool
from multiprocessing import Pipe, connection
from threading import Thread
import threading

import numpy as np

#from sdesolver.sdecallback import SDECallbackBase, SDECallbackCoeff, SDECallbackBoundary, SDECallbackSplit
#from sdesolver.loop.pyloop import py_integration_loop
#from sdesolver.util.datastructures import SDEPPStateOldstyle
#from sdesolver.supervisor import Supervisor
from .sdecallback import SDECallbackBase, SDECallbackCoeff, SDECallbackBoundary, SDECallbackSplit
from .loop.pyloop import py_integration_loop
from .util.datastructures import SDEPPStateOldstyle, SDEPseudoParticle
from .supervisor import Supervisor

class SDE:
    """
    This class is intended to contain the physical (i.e. non-numerical)
    aspects of an SDE. Those are:
    - the coefficients (drift, diffusion)
    - parameters for those
    - boundaries
    - initial condition (list of 2-tuples of t and [x])
    - number of dimensions of the problem (inferred from the initial condition)

    The coefficient and boundary callbacks can be set either by passing 
    callbacks to the constructor or by overriding the respective 
    functions when inheriting this class (or a combination of both).
    Both are automatically compiled to cfuncs if not done manually. This is
    done by the :py:class:`SDECallbackBoundary` and 
    :py:class:`SDECallbackCoeff` classes respectively. FIXME: Is this tested?
    It will probably not work because of the self parameter in the inherited
    functions.

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

    def __init__(self, initial_condition, drift=None, diffusion=None, boundary=None, split=None):
        if isinstance(initial_condition, int):
            raise ValueError("specifying ndim is deprecated")

        self._set_callback(drift, "drift", SDECallbackCoeff)
        self._set_callback(diffusion, "diffusion", SDECallbackCoeff)
        self._set_callback(boundary, "boundary", SDECallbackBoundary)
        self._set_callback(split, "split", SDECallbackSplit)
        
        self.initial_condition = initial_condition
        self.ndim = len(initial_condition[0][1])

        # check initial condition for consistency
        for t, x in initial_condition:
            assert len(x) == self.ndim

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

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, p):
        self.set_parameters(p)

    def set_parameters(self, parameters):
        """
        set all parameters with the same dict. Superflous parameters are
        ignored by SDECallbackBase.
        """
        self._parameters = parameters
        self.drift.parameters = parameters
        self.diffusion.parameters = parameters
        self.boundary.parameters = parameters
        self.split.parameters = parameters
    
    # doesnt work bc the callbacks extract their required parameters
    #def get_parameters(self):
    #    """
    #    gets the parameters set. Raises an exception if the callbacks
    #    have different parameter sets.
    #    """
    #    parameters = self.drift.parameters

    #    for cb in [self.diffusion, self.boundary, self.split]:
    #        if parameters != cb.parameters:
    #            for k, v in parameters.items():
    #                print(k, v)
    #            for k, v in cb.parameters.items():
    #                print(k, v)
    #            raise ValueError(f"Parameters of {cb} differ")

    #    return parameters
    
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

    def split(self, t, x, last_t, last_x, w):
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

    def copy(self, parameters=None, initial_condition=None, drift=None, diffusion=None, boundary=None, split=None):
        """
        Make a copy of this SDE, changing the properties that are given to this method
        """
        c = type(self)(self.initial_condition if initial_condition is None else initial_condition,
                       self.drift.pyfunc if drift is None else drift, 
                       self.diffusion.pyfunc if diffusion is None else diffusion, 
                       self.boundary.pyfunc if boundary is None else boundary,
                       self.split.pyfunc if split is None else split)
        c.set_parameters(self.parameters if parameters is None else parameters)
        return c

    def __str__(self):
        return f"SDE with parameters {self.parameters} and initial condition {self.initial_condition}"

class SDESolution:
    """
    Numerical solution of a stochastic differential equation.

    :param sde: The SDE that has been solved.
    :type sde: :py:class:`SDE`

    """
    def __init__(self, sde):
        self.sde = copy.copy(sde)
        self.observations = {}
        self.particle_count = 0
        self.observation_count = 0
        self._escaped_lists = {}
        #self._escaped_arrays = {}
        #self._escaped_updated = True

    def _add_observations(self, observation_times, positions, weights):
        try:
            assert len(observation_times) == len(positions) and len(positions) == len(weights)
        except AssertionError as e:
            print("obspos", len(observation_times), len(positions), len(weights))
            raise e

        self.particle_count += 1

        for t, x, w in list(zip(observation_times, positions.copy(), weights)):
            self.observation_count += 1
            if not t in self.observations:
                self.observations[t] = {'x': [x], 'weights': [w]}
            else:
                self.observations[t]['x'].append(x)
                self.observations[t]['weights'].append(w)

    def __getitem__(self, s):
        """
        Get the observed pseudo particles at one of the times.
        """
        if s not in self.observation_times:
            logging.warning(f"Key {s} not found in SDESolution, returning empty arrays...")
            return {'x': np.empty(0), 'weights': np.empty(0)}

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
        """
        Wrapper around the cython function :py:func:`py_integration_loop`
        Using a supervisor here reduces performance significantly (x2-x4)
        """
        observations_contiguous = np.empty(len(observation_times) * sde.ndim)
        this_weights = np.empty(len(observation_times))

        # these arrays are needed because we want to give the loop pointers
        t_array = np.empty(1, dtype=np.float64)
        weight_array = np.empty(1, dtype=np.float64)
        observation_count_array = np.empty(1, dtype=np.int32)

        t_array[0] = pp_t
        weight_array[0] = initial_weight

        split_times = []
        split_points = []
        split_weights = []

        start_cpp = time.perf_counter()
        boundary_state = py_integration_loop(observations_contiguous, observation_count_array, t_array, pp_x,
                                 sde.drift.address, sde.diffusion.address, sde.boundary.address, sde.split.address,
                                 seed, timestep, observation_times, split_times, split_points, split_weights, this_weights, weight_array, scheme)

        end_cpp = time.perf_counter()
        time_cpp = end_cpp - start_cpp

        if not supervisor_pipe is None:
            supervisor_pipe.send({'thread_id': threading.get_ident(), 'time_cpp': time_cpp, 'time_phys': t_array[0] - pp_t, 'split': len(split_times), 'particles' : 0})

        observations = observations_contiguous.reshape((-1, sde.ndim))[:observation_count_array[0]]
        this_weights = this_weights[:observation_count_array[0]]

        particle_info = {'final_t': t_array[0],
                         'final_x': pp_x,
                         'final_weight': weight_array[0],
                         'observations': observations,
                         'boundary_state': boundary_state,
                         'weights': this_weights
                        }

        split_info = list(zip(split_times, np.array(split_points), split_weights)) 

        # catch some cases that should not occur
        if not len(split_times) == 0:
            if np.min(split_times) <= pp_t:
                print(f"Particle back-splitted")
            if np.max(split_times) > np.max(observation_times):
                print(f"particle behind last observation time")

        return particle_info, split_info, time_cpp

    @staticmethod
    def solve_slice(sde, timestep, observation_times, scheme, seeds_slice, init, supervisor_pipe=None, skip_splits=False):
        """
        Solve a set (slice) of pseudo-particles using :py:func:`solve_one`.

        Parameters
        ----------
        sde : SDE
            The SDE to solve
        timestep : double
        observation_times : list of double
        scheme : str
            Name of the integration scheme to use
        seeds_slice : list of integers
            List of seeds, same length as init
        init : list of (double, [double], double)-tuples
            Initial time, position and weights of the slice's particles.
            Same length as seeds_slice
        supervisor_pipe : Connection

        """
        particles = []
        splits = []

        last_p = 0
        last_s = 0
        total_time_cpp = 0

        assert len(init) == len(seeds_slice)

        for (pp_t, pp_x, w), seed in zip(init, seeds_slice):
            # slice observation_times to avoid multiple observations of particles at their start time
            observation_times_slice = np.array([t for t in observation_times if t >= pp_t])
            pinfo, sinfo, time_cpp = SDESolver.solve_one(pp_t, pp_x, seed, observation_times_slice, sde, timestep, scheme, w)#supervisor_pipe)

            # crop observation_times_slice to account for escaped particles
            pinfo['observation_times_slice'] = observation_times_slice[:len(pinfo['observations'])]

            if skip_splits:
                # reset weights to start (if split particles will be ignored)
                pinfo['weights'] = np.array([w] * len(pinfo['observations']))
                pinfo['final_weight'] = w
            else:
                splits += sinfo

            particles.append(pinfo)
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

    def schedule_slices(self, sde, timestep, observation_times, seed_stream, init, pool, supervisor=None, sdesolution=None, particle_count_limit=np.inf):
        """
        Apply the slices of a set pseudo-particles to a thread pool, 
        using :py:func:`solve_one`. Returns a SDESolution instance
        containing the results.

        Since this function's call order (from splits) depends on the runtime
        of the worker threads, the seeding procedure used will probably lead 
        to non-deterministic behaviour even for fixed seeds. This could be
        fixed in the future by "inheriting" some kind of state from here to
        the recursive calls.

        Parameters
        ----------
        sde : SDE
            The SDE to solve
        timestep : double
        observation_times : list of double
        scheme : str
            Name of the integration scheme to use
        seed_stream : numpy Generator object
            Seed stream for generating the pseudo particle seeds
        init : list of (double, [double], double)-tuples
            Initial time, position and weights of the pseudo particles.
        pool : ThreadPool
            pool of threads to apply the workload to
        supervisor : supervisor.Supervisor (optional)

        """
        if sdesolution is None:
            sdesolution = SDESolution(sde)

        nthreads = pool._processes
        asyncresults = []

        # since this function's call order (from splits) depends on the runtime of the worker threads,
        # the seeding procedure used will probably lead to non-deterministic behaviour even for fixed seeds
        # this could be fixable by "inheriting" some kind of state from here to the recursive calls

        if len(init) == 0:
            return

        collected_split_starts = []
        
        skip_splits = sdesolution.particle_count >= particle_count_limit

        for mod in range(nthreads):
            if supervisor is not None:
                recv, send = Pipe(duplex=False)
                supervisor.attach(recv)
            else:
                recv, send = None, None

            init_slice = init[mod::nthreads]
            seeds = seed_stream.integers(sys.maxsize, size=len(init_slice))
            slice_args = (sde, timestep, observation_times, self.scheme,
                          seeds, init_slice, send, skip_splits)

            asr = pool.apply_async(self.solve_slice, slice_args)
            asyncresults.append(asr)

        for asr in asyncresults:
            particles, splits = asr.get()
            split_t = np.array([t for t, _, _ in splits])

            for par in particles:
                if par['boundary_state'] != 0:
                    sdesolution._add_escaped(par['final_t'], par['final_x'], par['final_weight'], par['boundary_state']) 

                sdesolution._add_observations(par['observation_times_slice'], par['observations'], par['weights'])

            collected_split_starts += splits

        self.schedule_slices(sde, timestep, observation_times, seed_stream, collected_split_starts, pool, supervisor=supervisor, sdesolution=sdesolution, particle_count_limit=particle_count_limit)

        # import pickle
        # with open("splits.pickle", mode="wb") as f:
        #     pickle.dump(collected_split_starts, f)

        return sdesolution

    def _format_time(self, t):
        return "{}s{}ms{}us".format(int(t), int(int(t * 1e3) % 1e3), int(int(t * 1e6) % 1e6))

    def solve(self, sde, timestep, observation_times, seed_stream=None, nthreads=4, supervise=True, particle_count_limit=np.inf):
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

        :param supervise: Use a Supervisor to print regular status messages about the calculation progress.
        :type supervise: bool
        """

        observation_times = np.array(sorted(observation_times))

        if seed_stream is None:
            from numpy.random import Generator, PCG64
            #seed_stream = Generator(PCG64(1234567890))
            seed_stream = Generator(PCG64(random.randint(0, int(1e10))))

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

        print("sde drift:", ", ".join([f"{k}: {v}" for k, v in sde.drift.parameters.items()]))
        print("sde diffusion:", ", ".join([f"{k}: {v}" for k, v in sde.diffusion.parameters.items()]))
        sdesolution = self.schedule_slices(sde, timestep, observation_times, seed_stream, init_copy, pool, supervisor=supervisor, particle_count_limit=particle_count_limit)

        end = time.perf_counter()

        if supervise:
            supervisor.interrupt()
            supervisor_thread.join()
            logging.info("Supervisor thread joined")

        total_time = end - start
        logging.info("Total simulation runtime: {}".format(self._format_time(total_time)))
        if supervise:
            time_cpp = supervisor.data[1]['time_cpp']
            logging.info("Total cpp runtime: {}".format(self._format_time(time_cpp)))
            logging.info("python overhead (= 1 - cpp/(nthreads * total)): {:.3g}%".format(100 * (1 - time_cpp / (nthreads * total_time))))


        return {'solution': sdesolution, 'runtime': total_time}

