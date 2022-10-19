from numba import njit, float64, boolean, prange
from numba.experimental import jitclass
from numba.typed import List
from numba.core.typing.asnumbatype import as_numba_type
import numpy as np
from numpy.random import SeedSequence, PCG64, Generator
import copy
import time

class SDE:
    def __init__(self, ndim, drift_callback=None, diffusion_callback=None, check_boundaries=None, initial_condition=None):
        if not drift_callback is None:
            self.drift_callback = drift_callback
        elif not type(self).drift is SDE.drift:
            self.drift_callback = self.drift
        else:
            raise NotImplementedError("Either drift_callback must be given on initialisation or drift must be overriden")

        if not diffusion_callback is None:
            self.diffusion_callback = diffusion_callback
        elif not type(self).diffusion is SDE.diffusion:
            self.diffusion_callback = self.diffusion
        else:
            raise NotImplementedError("Either diffusion_callback must be given on initialisation or diffusion must be overriden")

        if not check_boundaries is None:
            self.boundaries_callback = check_boundaries
        elif not type(self).check_boundaries is SDE.check_boundaries:
            self.boundaries_callback = self.check_boundaries
        else:
            raise NotImplementedError("Either boundaries_callback must be given on initialisation or boundaries must be overriden")

        self.initial_condition = initial_condition
        self.ndim = ndim

    def drift(self, t, x):
        """
        The drift term of the SDE model.
        It is passed a time and a position (an array [x0, x1,...])
        and should return a drift vector with the same spatial
        dimensionality
        """
        pass

    def diffusion(self, t, x):
        """
        The diffusion term of the SDE model.
        It is passed a time and a position (an array [x0, x1,...])
        and should return a diffusion matrix with the same spatial
        dimensionality
        """
        pass

    def check_boundaries(self, t, x):
        """
        function returning None or a string, depending on wether
        the particle hit a boundary (and which one in case it did)
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

def _solve_backend(pps, rngs, timestep, sde, scheme):
    for pp, rng in zip(pps, rngs):
        while not pp.finished:
            escaped = sde.boundaries_callback(pp.t, pp.x)
            if not escaped is None:
                pp.finished_reason = escaped
                pp.finished = True
                continue

            rndvec = rng.standard_normal(sde.ndim)
            new_p = scheme(pp.t, pp.x, rndvec, timestep, sde.drift_callback, sde.diffusion_callback)
            new_t, new_x = new_p.t, new_p.x

            pp.t = new_t
            pp.x = new_x

class SDEPPStateOldstyle:
    """ this class is for bodging this solver together with the C++ solver """
    def __init__(self, t, x, breakpoint_state):
        from pybatch.pybreakpointstate import PyBreakpointState

        self.t = t
        self.x = x

        if breakpoint_state == 'time':
            self.breakpoint_state = PyBreakpointState.TIME
        else:
            self.breakpoint_state = PyBreakpointState.NONE


class SDESolver:
    def __init__(self, scheme, noise_term=None):
        self.scheme = scheme
        self.noise_term = noise_term
        if not noise_term is None:
            print("WARNING: noise_term is currently ignored")


    def solve(self, sde, timestep):
        pps = []
        for init_pp in sde.initial_condition:
            pps.append(init_pp.copy())

        #pps = np.array(pps)
        seeds = SeedSequence(1234)
        # maybe switch to PCG64DXSM (https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html)
        rngs = [Generator(PCG64(s)) for s in seeds.spawn(len(pps))] 

        start = time.perf_counter()
        _solve_backend(pps, rngs, timestep, sde, self.scheme)
        end = time.perf_counter()
        print("Elapsed backend = {}us".format((end - start) * 1e6))
        return pps

    def solve_oldstyle(self, sde, timestep):
        pps = self.solve(sde, timestep)
        return SDESolver.get_oldstyle_like_states(pps)

    @staticmethod
    def get_oldstyle_like_states(newstyle_pps):
        pstates = []
        for pp in newstyle_pps:
            pstates.append(SDEPPStateOldstyle(pp.t, np.array([pp.x]).T, pp.finished_reason))

        return pstates



def sde_scheme_euler(pp, rndvec, timestep, drift, diffusion):
    t, x = pp.t, pp.x
    t_new = t + timestep
    #drift_term = timestep * drift(t, x)
    #diff_term = np.dot(diffusion(t, x), rndvec)
    x_new = x + timestep * drift(t, x) + np.dot(diffusion(t, x), rndvec) * np.sqrt(timestep)
    #print(x_new, drift_term, diff_term)
    return t_new, x_new
