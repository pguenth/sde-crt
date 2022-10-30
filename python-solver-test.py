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

from numba import njit, f8, cfunc, types, carray

import inspect 
import logging

logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

#@njit(f8[:](f8, f8[:]))
@cfunc(f8[:](f8, f8[:]), cache=True)
def drift(t, x):
    # cpp: kruells_shockaccel2_drift_94_2
    Xsh = 0.001215
    a = 0.0375
    b = 0.0225
    k_syn = 0
    q = 1

    #v0 = kruells94_dkappadx_dep(x[0], Xsh, a, b, q) + kruells94_beta(x[0], Xsh, a, b)
    #v1 = - (x[1]) * (kruells94_dbetadx(x[0], Xsh, b) / 3 + k_syn * x[1])
    #return np.array([v0, v1])
    return np.array([0.0, 1.0])

#@cfunc(types.void(types.CPointer(types.double), types.double, types.CPointer(types.double)), cache=True)
@cfunc_coeff
def drift_test(out, t, x):
    # cpp: kruells_shockaccel2_drift_94_2
    Xsh = 0.001215
    a = 0.0375
    b = 0.0225
    k_syn = 0
    q = 1

    x_a = carray(x, (2,))
    v0 = kruells94_dkappadx_dep(x_a[0], Xsh, a, b, q) + kruells94_beta(x_a[0], Xsh, a, b)
    v1 = - (x_a[1]) * (kruells94_dbetadx(x_a[0], Xsh, b) / 3 + k_syn * x_a[1])

    out_a = carray(out, (1,))
    out_a[0] = v0
    out_a[1] = v1
    return

#@cfunc(types.void(types.CPointer(types.double), types.double, types.CPointer(types.double)), cache=True)
@cfunc_coeff
def diffusion_test(out, t, x):
    # cpp: kruells_shockaccel2_diffusion
    Xsh = 0.001215
    a = 0.0375
    b = 0.0225
    q = 1
    x_a = carray(x, (2,))
    diffval = np.sqrt(2.0 * kruells94_kappa_dep(x[0], Xsh, a, b, q))

    out_a = carray(out, (2, 2))
    out_a[0, 0] = diffval
    out_a[1, 0] = 0
    out_a[0, 1] = 0
    out_a[1, 1] = 0
    return

#@njit(f8[:,:](f8, f8[:]))
@cfunc(f8[:,:](f8, f8[:]), cache=True)
def diffusion(t, x):
    # cpp: kruells_shockaccel2_diffusion
    Xsh = 0.001215
    a = 0.0375
    b = 0.0225
    q = 1
    diffval = np.sqrt(2.0 * kruells94_kappa_dep(x[0], Xsh, a, b, q))
    return diffval * np.array([[1.0, 0.0], [0.0, 0.0]])

#@njit(boolean(f8, f8[:]))
#@cfunc(types.int32(types.double, types.CPointer(types.double)), cache=True)
@cfunc_boundary
def boundaries(t, x):
    ###
    return 0
    ###

    x_a = carray(x, (2,))
    if np.abs(x_a[0]) > 0.005:
        return 1
    else:
        return 0


print("compiled numba cfuncs")


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

        solution = self._solver.solve(self._sde, self._params['dt'], [self._params['Tmax']])

        end = time.perf_counter()
        print("finished timing")
        print("Elapsed = {}us".format((end - start) * 1e6))
        self._states = solution.get_oldstyle_pps(self._params['Tmax'])
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

#from src.scheme import sde_scheme_euler_cython, sde_scheme_euler_cpp
cachedir = "cache"
figdir = "figures"
def kruells9a1_newstyle():
    name = inspect.currentframe().f_code.co_name

    T = 20.0
    t_inj = 0.001
    dt = 0.001
    n_particle = int(T / t_inj)

    init = [SDEPseudoParticle(i * t_inj, np.array([0.0, 1.0])) for i in range(n_particle)]
    sde = SDE(2, init, drift_test, diffusion_test, boundaries)
    sdesolver = SDESolver(b'euler')

    param = { 'sde' : sde,
              'solver' : sdesolver,
              'dt' : dt,
            }

    #times = np.array([0.64, 2.0, 6.4, 20, 200])
    times = np.array([T])

    cache = None#PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(NumpyBatch, cache, param, times, confine_x=0.05, bin_count=30)
    #histosetx.map_tree(lambda b : b.set(nthreads=8), "batch")

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    #nfig.add(powerlaw, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')



kruells9a1_newstyle()
#import cProfile
#pr = cProfile.Profile()
#cProfile.run('kruells9a1_newstyle()', filename="test-cpp-propagation.perf")
