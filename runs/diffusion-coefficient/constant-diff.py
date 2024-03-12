import sdesolver as sdes

import inspect
import time

from grapheval.nodefigure import NodeFigure, NodeFigureFormat
from grapheval.cache import PickleNodeCache
from grapheval.node import *
from grapheval.graph import draw_node_chain

from src.basicnodes import *
from src.newnodes import *

from scipy import integrate
from scipy.optimize import root_scalar

import formats

from numba import njit, f8, carray
import numba

import inspect 
import logging
import ctypes
import warnings

#warnings.simplefilter("error", np.VisibleDeprecationWarning)


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

def drift_const(out, t, x, Xsh, a, b, k_syn):
    v0 = kruells94_beta(x[0], Xsh, a, b)
    v1 = - (x[1]) * (kruells94_dbetadx(x[0], Xsh, b) / 3 + k_syn * x[1])

    out[0] = v0
    out[1] = v1

def diffusion_const(out, t, x, kappa):
    out_a = carray(out, (2, 2))
    out_a[0, 0] = np.sqrt(2.0 * kappa)
    out_a[1, 0] = 0
    out_a[0, 1] = 0
    out_a[1, 1] = 0

def boundaries(t, x):
    return 0

def nosplit(t, x, last_t, last_x, w):
    return False

def split(t, x, last_t, last_x, w, condition):
    if x[1] / last_x[1] >= condition:#1.8:#1.41:
        return True
    else:
        return False


cachedir = "cache"
figdir = "figures"

name = "const-kappa-test"


T = 200.0
t_inj = 0.05
# T = 40.0
# t_inj = 0.005
x0 = np.array([0.0, 1.0])
dt = 0.001
confine_x=100
n_particle = int(T / t_inj) 

init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]
# init = [(0.0, np.copy(x0)) for i in range(n_particle)]

parameters = {
        'Xsh' : 0.001215, 
        'a' : 0.0375,
        'b' : 0.0225,
        'k_syn' : 0,
        'q' : 1,
        'kappa': 0.001
    }

sde = sdes.SDE(init, drift, diffusion, boundaries, split)
sde_const = sdes.SDE(init, drift_const, diffusion_const, boundaries, split)

sde.set_parameters(parameters | {'condition': 1.6})
sde_const.set_parameters(parameters | {'condition': 2.2})

cache = PickleNodeCache(cachedir, name)

obs_at = [T/8, T/4, T / 2, T]
solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=64, cache=cache, kconst=False)
solvernode_const = SDESolverNode('solver_const', sde=sde_const, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=64, cache=cache, kconst=True)

histo_opts = {'bin_count' : 30, 'plot' : True, 'cache' : cache, 'ignore_cache' : False, 'label': 'T={T}, kappa is const? {kconst}'}

valuesx = SDEValuesNode(f'valuesx', {'x' : solvernode['solution'][T]['x'], 'weights': solvernode['solution'][T]['weights']}, index=0, T=T, cache=cache)
valuesp = SDEValuesNode(f'valuesp', {'x' : solvernode['solution'][T]['x'], 'weights': solvernode['solution'][T]['weights']}, index=1, T=T, cache=cache,
    confine_range=[(0, -confine_x, confine_x)],
)
histogramx = HistogramNode(f'histox', {'values' : valuesx['values'], 'weights' : valuesx['weights']}, log_bins=False, normalize='width', **histo_opts)
histogramp = HistogramNode(f'histop', {'values' : valuesp['values'], 'weights' : valuesp['weights']}, log_bins=True, normalize='density', **histo_opts)
pl = MLEPowerlawNode('pl', {'values' : valuesp['values'], 'weights' : valuesp['weights']}, plot=True, cache=cache)


valuesx_const = SDEValuesNode(f'valuesx_const', {'x' : solvernode_const['solution'][T]['x'], 'weights': solvernode_const['solution'][T]['weights']}, index=0, T=T, cache=cache)
valuesp_const = SDEValuesNode(f'valuesp_const', {'x' : solvernode_const['solution'][T]['x'], 'weights': solvernode_const['solution'][T]['weights']}, index=1, T=T, cache=cache,
    confine_range=[(0, -confine_x, confine_x)],
)
histogramx_const = HistogramNode(f'histox_const', {'values' : valuesx_const['values'], 'weights' : valuesx_const['weights']}, log_bins=False, normalize='width', **histo_opts)
histogramp_const = HistogramNode(f'histop_const', {'values' : valuesp_const['values'], 'weights' : valuesp_const['weights']}, log_bins=True, normalize='density', **histo_opts)
pl_const = MLEPowerlawNode('pl_const', {'values' : valuesp_const['values'], 'weights' : valuesp_const['weights']}, plot=True, cache=cache)

nfig = NodeFigure(formats.doublehist)
nfig.add(histogramx, 0)
nfig.add(histogramx_const, 0)
nfig.add(histogramp, 1)
nfig.add(pl, 1)
nfig.add(histogramp_const, 1)
nfig.add(pl_const, 1)
nfig.savefig(figdir + '/' + name + '.pdf')

