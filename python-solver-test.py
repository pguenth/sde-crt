import sdesolver as sdes

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
import numba

import inspect 
import logging
import ctypes

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

#def cfunc_ndarray(params):
#    list(inspect.signature(self._pyfunc).parameters.keys())
#    def deco(func):



    
# test accessing a numpy array in a coefficient function
def drift_map_dir(out, t, x, addr):
    arr = carray(sdes.address_as_void_pointer(addr), 5, dtype=np.float64)
    v0 = kruells94_dkappadx_dep(x[0], arr[0], arr[1], arr[2], arr[4]) + kruells94_beta(x[0], arr[0], arr[1], arr[2])
    v1 = - (x[1]) * (kruells94_dbetadx(x[0], arr[0], arr[2]) / 3 + arr[3] * x[1])

    out[0] = v0
    out[1] = v1

# test accessing a numpy array in a coefficient function
def drift_map(out, t, x, addr):
    arr = carray(address_as_void_pointer(addr), 5, dtype=np.float64)
    v0 = kruells94_dkappadx_dep(x[0], arr[0], arr[1], arr[2], arr[4]) + kruells94_beta(x[0], arr[0], arr[1], arr[2])
    v1 = - (x[1]) * (kruells94_dbetadx(x[0], arr[0], arr[2]) / 3 + arr[3] * x[1])

    out[0] = v0
    out[1] = v1

def drift(out, t, x, Xsh, a, b, k_syn, q):
    # cpp: kruells_shockaccel2_drift_94_2

    v0 = kruells94_dkappadx_dep(x[0], Xsh, a, b, q) + kruells94_beta(x[0], Xsh, a, b)
    v1 = - (x[1]) * (kruells94_dbetadx(x[0], Xsh, b) / 3 + k_syn * x[1])

    out[0] = v0
    out[1] = v1

def diffusion(out, t, x, Xsh, a, b, q):
    # cpp: kruells_shockaccel2_diffusion
    diffval = np.sqrt(2.0 * kruells94_kappa_dep(x[0], Xsh, a, b, q))

    # here carray is required to reshape the contiguous pointer
    out_a = carray(out, (2, 2))
    out_a[0, 0] = diffval
    out_a[1, 0] = 0
    out_a[0, 1] = 0
    out_a[1, 1] = 0

def boundaries(t, x):
    ###
    return 0
    ###

    x_a = carray(x, (2,))
    if np.abs(x_a[0]) > 0.005:
        return 1
    else:
        return 0

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

    T = 20.0
    t_inj = 0.01
    dt = 0.001
    n_particle = int(T / t_inj)

    x0 = np.array([0.0, 1.0])
    init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]
   
    arr = np.array([0.001215, 0.0375, 0.0225, 0, 1])

    parameters = {
            'Xsh' : 0.001215, 
            'a' : 0.0375,
            'b' : 0.0225,
            'k_syn' : 0,
            'q' : 1,
            'addr' : arr.ctypes.data,#_as(ctypes.POINTER(ctypes.c_double)),
            'arr' : arr
        }
    drift_map_cb = sdes.SDECallbackCoeff(drift_map_dir, parameter_types={'arr': types.double[:], 'addr': types.int64})
    sde = sdes.SDE(2, init, drift_map_cb, diffusion, boundaries)
    sde.set_parameters(parameters)
    sdesolver = sdes.SDESolver(b'euler')

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
