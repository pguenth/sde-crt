from numba import njit, float64, boolean, prange, types, cfunc
import ctypes
from numba.experimental import jitclass
from numba.typed import List
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.ccallback import CFunc
import numpy as np
from numpy.random import SeedSequence, PCG64, Generator
import copy
import time
import inspect
import logging

from src.c.pyloop import pyploop, print_double

from numba.extending import get_cython_function_address

print_double_addr = get_cython_function_address("src.c.pyloop", "print_double")
print_double_numba = ctypes.CFUNCTYPE(None, ctypes.c_double)(print_double_addr)


def _cfunc_sde_base(types_base, type_return, optional_func=None, param_types=None, **kwargs):
    """
    a convenience decorator for compiling coefficient functions (drift/diffusion)
    to numba cfuncs callable from C++ code. In any case, the first three arguments
    are `out, t, x`. `t` and `x` are the time and phase-space point at which
    the function should evaluate; `out` is an array where the result should be
    written to. Those three are automatically typed by this decorator.

    The remaining arguments (referred to as 'parameters' further on) are by
    default typed as double when calling this decorator without arguments:

        @cfunc_coeff
        def example(out, t, x, p0, p1, p2):
            pass

    Using arguments, another type can be chosen:

        @cfunc_coeff(param_types=[types.int32, types.int32. types.double])
        def example(out, t, x, an_int, another_int, a_double):
            pass

    Keyword arguments are passed through to numba.cfunc()
    """
    
    def deco(func):
        if param_types is None:
            pcount = len(inspect.signature(func).parameters) - len(types_base)
            ptypes = [types.double] * pcount
        else:
            ptypes = param_types

        return cfunc(type_return(*types_base, *ptypes), **kwargs)(func)

    if not optional_func is None:
        return deco(optional_func)
    else:
        return deco

def cfunc_coeff(*args, **kwargs):
    arg_base = (types.CPointer(types.double), types.double, types.CPointer(types.double))
    return _cfunc_sde_base(arg_base, types.void, *args, **kwargs)

def cfunc_boundary(*args, **kwargs):
    arg_base = (types.double, types.CPointer(types.double))
    return _cfunc_sde_base(arg_base, types.int32, *args, **kwargs)



class SDE:
    """
    This class is intended to contain every physical (non-numerical)
    aspect of an SDE.
    """
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
        if not type(cback) is CFunc:
            logging.warning("Callback for {} is not a numba-compiled C function. Trying to compile with default types.")
            cback = decorator(cback)

        # set the resulting callback
        setattr(self, name, cback)
        setattr(self, "_noparam_" + name, cback)

    def _set_parameters_of(self, name, parameters, cbtype):
        if parameters is None:
            setattr(self, name, getattr(self, "_noparam_" + name))
        else:
            orig = getattr(self, "_noparam_" + name)
            if cbtype == "coeff":
                newf = cfunc_coeff(lambda out, t, x : orig(out, t, x, *parameters))
            elif cbtype == "boundary":
                newf = cfunc_boundary(lambda t, x : orig(t, x, *parameters))
            else:
                raise ValueError("cbtype must be either coeff or boundary")

            setattr(self, name, newf)

    def __init__(self, ndim, initial_condition=None, drift=None, diffusion=None, boundary=None):
        self._set_callback(drift, "drift", cfunc_coeff)
        self._set_callback(diffusion, "diffusion", cfunc_coeff)
        self._set_callback(boundary, "boundary", cfunc_boundary)

        self.initial_condition = initial_condition
        self.ndim = ndim

    def set_parameters(self, drift_parameters=None, diffusion_parameters=None, boundary_parameters=None):
        """
        set all or a subset of parameters at once
        """
        if not drift_parameters is None:
            self.drift_parameters = drift_parameters

        if not diffusion_parameters is None:
            self.diffusion_parameters = diffusion_parameters

        if not boundary_parameters is None:
            self.boundary_parameters = boundary_parameters
    
    @property
    def drift_parameters(self):
        """
        don't write to this
        eventual solution: move callback into own class and move callback parameters into own class.
        the latter then can provide custom item access and therefore can instruct recompilation
        on single-parameter changes. currently not worth the effort, but would be cool
        additionally, this would remove some repeated code from here
        """
        return self._drift_parameters

    @drift_parameters.setter
    def drift_parameters(self, p):
        self._drift_parameters = p
        self._set_parameters_of("drift", p, "coeff")
    
    @property
    def diffusion_parameters(self):
        """
        see drift_parameters
        """
        return self._diffusion_parameters

    @diffusion_parameters.setter
    def diffusion_parameters(self, p):
        self._diffusion_parameters = p
        self._set_parameters_of("diffusion", p, "coeff")
    
    @property
    def boundary_parameters(self):
        """
        see drift_parameters
        """
        return self._boundary_parameters

    @boundary_parameters.setter
    def boundary_parameters(self, p):
        self._boundary_parameters = p
        self._set_parameters_of("boundary", p, "boundary")
        

    def drift(self, out, t, x):
        """
        The drift term of the SDE model.
        It is passed a time and a position (an array [x0, x1,...])
        and should write a drift vector with the same spatial
        dimensionality to out
        """
        pass

    def diffusion(self, out, t, x):
        """
        The diffusion term of the SDE model.
        It is passed a time and a position (an array [x0, x1,...])
        and should write a diffusion matrix with the same spatial
        dimensionality to out
        """
        pass

    def boundary(self, t, x):
        """
        function returning an int, depending on wether
        the particle hit a boundary. Must return 0 if no boundary is
        reached.
        """
        pass


    

class SDEPseudoParticle:
    finished_reason : str
    def __init__(self, t0, x0, finished=False, finished_reason='none'):
        self.t = t0
        self.x = x0
        self.finished = finished
        self.finished_reason = finished_reason

    def __deepcopy__(self, memo):
        return self.copy()

    def copy(self):
        return SDEPseudoParticle(self.t, self.x, self.finished, self.finished_reason)


class SDEPPStateOldstyle:
    """ this class is for bodging this solver together with the C++ solver """
    def __init__(self, t, x, breakpoint_state):
        from pybatch.pybreakpointstate import PyBreakpointState

        self.t = t
        self.x = x

        if breakpoint_state == 0:
            self.breakpoint_state = PyBreakpointState.TIME
        else:
            self.breakpoint_state = PyBreakpointState.NONE


class SDESolution:
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
        #for k in self._escaped_lists.keys():
        #    self._escaped_arrays[k]['t'] = np.array(self._escaped[k]['t'])
        #    self._escaped_arrays[k]['x'] = np.array(self._escaped[k]['x'])

        #self._escaped_updated = False
        #return self._escaped_arrays
        return self._escaped_lists

    @property
    def observation_times(self):
        return self.observations.keys()

    @property
    def boundary_states(self):
        return self._escaped_lists.keys()


    def get_oldstyle_pps(self, observation_time):
        pstates = []
        for pp in self[observation_time]:
            pstates.append(SDEPPStateOldstyle(observation_time, np.array([pp]).T, 0))

        return pstates



class SDESolver:
    def __init__(self, scheme, noise_term=None):
        self.scheme = scheme
        self.noise_term = noise_term
        if not noise_term is None:
            print("WARNING: noise_term is currently ignored")

    def solve(self, sde, timestep, observation_times):
        res = SDESolution(sde)

        observation_times = np.array(observation_times)
        seeds = list(range(len(sde.initial_condition)))

        start = time.perf_counter()
        time_cpp = 0

        observations_contiguous = np.empty(len(observation_times) * sde.ndim)
        # these arrays are needed because we want to give the loop pointers
        t_array = np.empty(1, dtype=np.float64)
        observation_count_array = np.empty(1, dtype=np.int32)
        for pp, seed in zip(sde.initial_condition, seeds):
            start_cpp = time.perf_counter()
            t_array[0] = pp.t

            boundary_state = pyploop(observations_contiguous, observation_count_array, t_array, pp.x,
                                     sde.drift.address, sde.diffusion.address, sde.boundary.address,
                                     #ctypes.cast(sde.drift, ctypes.c_void_p).value, ctypes.cast(sde.diffusion, ctypes.c_void_p).value, ctypes.cast(sde.boundary, ctypes.c_void_p).value,
                                     seed, timestep, 
                                     observation_times, self.scheme)

            end_cpp = time.perf_counter()
            time_cpp += end_cpp - start_cpp

            if boundary_state != 0:
                res._add_escaped(t_array[0], pp.x, boundary_state)

            res._add_observations(observation_times, observations_contiguous.reshape((-1, sde.ndim)), observation_count_array[0])

        end = time.perf_counter()
        logging.info("Runtime for pseudoparticle propagation: {}us".format(time_cpp * 1e6))
        logging.info("Runtime for propagation and transformation: {}us".format((end - start) * 1e6))

        return res

    def solve_oldstyle(self, sde, timestep):
        pps = self.solve(sde, timestep)
        return SDESolver.get_oldstyle_like_states(pps)



#def sde_scheme_euler(t, x, rndvec, timestep, drift, diffusion):
#    #t, x = pp.t, pp.x
#    t_new = t + timestep
#    #drift_term = timestep * drift(t, x)
#    #diff_term = np.dot(diffusion(t, x), rndvec)
#    x_new = x + timestep * drift(t, x) + np.dot(diffusion(t, x), rndvec) * np.sqrt(timestep)
#    #print(x_new, drift_term, diff_term)
#    return t_new, x_new
