from src.sdesolver import *

import inspect
import time

from grapheval.nodefigure import NodeFigure, NodeFigureFormat
from grapheval.cache import PickleNodeCache
from grapheval.node import *
from grapheval.graph import draw_node_chain

from src.specialnodes import *

import chains
import formats

from numba import njit, f8

@njit(f8(f8, f8, f8, f8))
def kruells94_beta(x, Xsh, a, b):
    return a - b * np.tanh(x / Xsh)

@njit(f8(f8, f8, f8))
def kruells94_dbetadx(x, Xsh, b):
    return - b / (Xsh * np.cosh(x / Xsh)**2)

@njit(f8(f8, f8, f8, f8, f8))
def kruells94_kappa_dep(x, Xsh, a, b, q):
    return q * kruells94_beta(x, Xsh, a, b)**2

@njit(f8(f8, f8, f8, f8, f8))
def kruells94_dkappadx_dep(x, Xsh, a, b, q):
    return 2 * q * kruells94_beta(x, Xsh, a, b) * kruells94_dbetadx(x, Xsh, b)

@njit(f8[:](f8, f8[:]))
def drift(t, x):
    # cpp: kruells_shockaccel2_drift_94_2
    Xsh = 0.001215
    a = 0.0375
    b = 0.0225
    k_syn = 0
    q = 1

    v0 = kruells94_dkappadx_dep(x[0], Xsh, a, b, q) + kruells94_beta(x[0], Xsh, a, b)
    v1 = - (x[1]) * (kruells94_dbetadx(x[0], Xsh, b) / 3 + k_syn * x[1])
    return np.array([v0, v1])

@njit(f8[:,:](f8, f8[:]))
def diffusion(t, x):
    # cpp: kruells_shockaccel2_diffusion
    Xsh = 0.001215
    a = 0.0375
    b = 0.0225
    q = 1
    diffval = np.sqrt(2.0 * kruells94_kappa_dep(x[0], Xsh, a, b, q))
    return diffval * np.array([[1.0, 0.0], [0.0, 0.0]])

#@njit(boolean(f8, f8[:]))
def boundaries(t, x):
    if t > 0.2:
        return 'time'
    else:
        return None



# this class is for testing purposes only
class NumpyBatch:
    def __init__(self, params):
        self._sde = params['sde']
        self._solver = params['solver']

        self._states = None
        self._reconstructed = False
        self._params = params

    def run(self, nthreads=None):
        if not nthreads is None:
            print("warning: nthreads is currently ignored")

        print("start timing")
        start = time.perf_counter()
        pps = self._solver.solve(self._sde, 0.001)
        end = time.perf_counter()
        print("finished timing")
        print("Elapsed = {}us".format((end - start) * 1e6))
        #self._states = np.array([SDESolver.get_oldstyle_like_states(pps)]).T
        self._states = SDESolver.get_oldstyle_like_states(pps)
        return len(self._states)

    def step_all(self, steps=1):
        raise NotImplementedError("use run")

    @property
    def unfinished_count(self):
        raise NotImplementedError("partial solving deprecated")
    
    def _raise_reconstructed_exception(self):
        if self._reconstructed is True:
            raise ValueError("Cannot access C++ level functions on reconstructed objects")

    def state(self, index):
        return self._states[index]

    @property
    def states(self):
        if self._states is None:
            self.run()
        return self._states

    @property
    def integrator_values(self):
        return np.array([])

    # pickling doesnt create a new batch with full functionality
    # it only restores the states and integrator values
    def __reduce__(self):
        if type(self) == PyPseudoParticleBatch:
            raise ValueError("Cannot __reduce__ a PyPseudoParticleBatch (you must inherit from this class)")

        print("Reducing...")
        return type(self)._reconstruct, (self._params, self.states)

    @classmethod
    def _reconstruct(cls, params, states):
        instance = cls(params)
        instance._states = states
        return instance

cachedir = "cache"
figdir = "figures"
def kruells9a1_newstyle():
    name = inspect.currentframe().f_code.co_name
    init = [SDEPseudoParticle(i * 0.01, np.array([0.0, 1.0])) for i in range(20)]
    sde = SDE(2, drift, diffusion, boundaries, init)
    sdesolver = SDESolver(sde_scheme_euler)
    start = time.perf_counter()
    sdesolver.solve(sde, 0.001)
    end = time.perf_counter()
    print("Elapsed = {}us".format((end - start) * 1e6))
    exit()

    param = { 'sde' : sde,
              'solver' : sdesolver,
            }

    #times = np.array([0.64, 2.0, 6.4, 20, 200])
    times = np.array([0.2])

    cache = None#PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(NumpyBatch, cache, param, times, confine_x=0.05, bin_count=30)
    histosetx.map_tree(lambda b : b.set(nthreads=8), "batch")

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    #nfig.add(histosetp, 1)
    #nfig.add(powerlaw, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')


kruells9a1_newstyle()
#import cProfile
#pr = cProfile.Profile()
#cProfile.run('kruells9a1_newstyle()', filename="test-numba-math.perf")
