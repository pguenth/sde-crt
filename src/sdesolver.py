from numba import njit, float64, boolean, prange
from numba.experimental import jitclass
from numba.typed import List
from numba.core.typing.asnumbatype import as_numba_type
import numpy as np
from numpy.random import SeedSequence, PCG64, Generator
import copy

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

            

@jitclass([("t", float64), ("x", float64[:]), ("finished", boolean)])
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

#@jitclass([("next_double", as_numba_type(PCG64(0).cffi.next_double)), ("state", as_numba_type(PCG64(0).cffi.state_address))])
#class CRandomNumberGenerator:
#    def __init__(self, next_double, state_address):
#        self.next_double = next_double
#        self.state_address = state_address


@njit
def _normals(n, next_double, state):
    """
    from https://numpy.org/devdocs/reference/random/examples/numba.html
    no idea why this works, but it does
    """
    out = np.empty(n)
    for i in range((n + 1) // 2):
        x1 = 2.0 * next_double(state) - 1.0
        x2 = 2.0 * next_double(state) - 1.0
        r2 = x1 * x1 + x2 * x2
        while r2 >= 1.0 or r2 == 0.0:
            x1 = 2.0 * next_double(state) - 1.0
            x2 = 2.0 * next_double(state) - 1.0
            r2 = x1 * x1 + x2 * x2
        f = np.sqrt(-2.0 * np.log(r2) / r2)
        out[2 * i] = f * x1
        if 2 * i + 1 < n:
            out[2 * i + 1] = f * x2
    return out

@njit(parallel=True)
def _solve_backend(pps, rng_next_doubles, rng_states, timestep, sde_drift, sde_diff, sde_bound, sde_ndim, scheme):
    for i in prange(len(pps)):
        pp = pps[i]
        rng_state = rng_states[i]
        rng_next_double = rng_next_doubles[i]
        while not pp.finished:
            escaped = sde_bound(pp.t, pp.x)
            if not escaped is None:
                pp.finished_reason = escaped
                pp.finished = True
                continue

            rndvec = _normals(sde_ndim, rng_next_double, rng_state)
            new_t, new_x = scheme(pp, rndvec, timestep, sde_drift, sde_diff)

            pp.t = new_t
            pp.x = new_x



class SDESolver:
    def __init__(self, scheme, noise_term=None):
        self.scheme = scheme
        self.noise_term = noise_term
        if not noise_term is None:
            print("WARNING: noise_term is currently ignored")


    def solve(self, sde, timestep):
        pps = List()
        for init_pp in sde.initial_condition:
            pps.append(init_pp.copy())

        #pps = np.array(pps)
        seeds = SeedSequence(1234)
        # maybe switch to PCG64DXSM (https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html)
        bit_gens = [PCG64(s) for s in seeds.spawn(len(pps))]

        rng_next_doubles = [r.cffi.next_double for r in bit_gens]
        rng_states = List([r.cffi.state_address for r in bit_gens])
        #rngs = [CRandomNumberGenerator(r.cffi.next_double, r.cffi.state_address) for r in bit_gens]

        _solve_backend(pps, rng_next_doubles, rng_states, timestep, sde.drift_callback, sde.diffusion_callback, sde.boundaries_callback, sde.ndim, self.scheme)
        return list(pps)

@njit
def sde_scheme_euler(pp, rndvec, timestep, drift, diffusion):
    t, x = pp.t, pp.x
    t_new = t + timestep
    #drift_term = timestep * drift(t, x)
    #diff_term = np.dot(diffusion(t, x), rndvec)
    x_new = x + timestep * drift(t, x) + np.dot(diffusion(t, x), rndvec) * np.sqrt(timestep)
    #print(x_new, drift_term, diff_term)
    return t_new, x_new
