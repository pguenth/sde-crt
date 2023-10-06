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

def drift(out, t, x, a2, k_syn):
    # cpp: kruells_2ndorder_drift
    out[0] = 0
    out[1] = x[1] * (4 * a2 - k_syn * x[1])

def diffusion(out, t, x, kappa, a2):
    # cpp: kruells_2ndorder_diffusion

    # here carray is required to reshape the contiguous pointer
    out_a = carray(out, (2, 2))
    out_a[0, 0] = np.sqrt(2.0 * kappa)
    out_a[1, 0] = 0
    out_a[0, 1] = 0
    out_a[1, 1] = np.sqrt(2.0 * a2) * x[1]

def boundaries(t, x, L, Lyupper, Lylower):
    x_a = carray(x, (2,))
    if np.abs(x_a[0]) > L:
        return 1
    elif x_a[1] > Lyupper:
        return 2
    elif x_a[1] < Lylower:
        return 3
    else:
        return 0

def nosplit(t, x, last_t, last_x, w):
    return False

def split(t, x, last_t, last_x, w):
    if x[1] / last_x[1] >= 1.8:#1.41:
        return True
    else:
        return False


cachedir = "cache"
figdir = "figures"

name = "sa"

T = 640.0
t_inj = 0.001#0.0004
x0 = np.array([0.0, 1.0])
dt = 0.004
confine_x=1
n_particle = int(T / t_inj) 

init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]

parameters = { 'kappa' : 1,
          'a2' : 0.01,
          'k_syn' : 0,
          'L' : 20,
          'Lylower' : 0,
          'Lyupper' : 200,
        }

sde = sdes.SDE(2, init, drift, diffusion, boundaries, nosplit)
sde.set_parameters(parameters)

cache = PickleNodeCache(cachedir, name)

obs_at = [T/40, T / 10, T]
solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=64, cache=cache, splitted=False)

histo_opts = {'bin_count' : 100, 'plot' : True, 'cache' : cache, 'ignore_cache' : False, 'label': 'T={T}, splitted: {splitted}'}

valuesx = {}
valuesp = {}
histogramx = {}
histogramp = {}
for T_ in obs_at:
    valuesx[T_] = SDEValuesNode(f'valuesx_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=0, T=T_, cache=cache)
    valuesp[T_] = SDEValuesNode(f'valuesp_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=1, T=T_, cache=cache,
        confine_range=[(0, -confine_x, confine_x)],
    )
    histogramx[T_] = HistogramNode(f'histox_{T_}', {'values' : valuesx[T_]['values'], 'weights' : valuesx[T_]['weights']}, log_bins=False, normalize='width', **histo_opts)
    histogramp[T_] = HistogramNode(f'histop_{T_}', {'values' : valuesp[T_]['values'], 'weights' : valuesp[T_]['weights']}, log_bins=True, normalize='density', **histo_opts)

histograms_x = NodeGroup('gx', histogramx)
histograms_p = NodeGroup('gp', histogramp)


nfig = NodeFigure(formats.doublehist)
nfig.add(histograms_x, 0)
nfig.add(histograms_p, 1)
nfig.savefig(figdir + '/' + name + '.pdf')
