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

"""
Reproduces kruellsB1c (schlumpfhüte) when confining to the shock position is active.
"""

#warnings.simplefilter("error", np.VisibleDeprecationWarning)


logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@njit(f8(f8, f8, f8, f8))
def kruells94_beta(x, Xsh, a, b):
    return a - b * np.tanh(x / Xsh)

@njit(f8(f8, f8, f8))
def kruells94_dbetadx(x, Xsh, b):
    return - b / (Xsh * np.cosh(x / Xsh)**2)

def drift(out, t, x, Xsh, a, b, k_syn, q):
    # cpp: kruells_shockaccel2_drift_94_2

    v0 = kruells94_beta(x[0], Xsh, a, b)
    v1 = - (x[1]) * (kruells94_dbetadx(x[0], Xsh, b) / 3 + k_syn * x[1])

    out[0] = v0
    out[1] = v1

def diffusion(out, t, x, Xsh, a, b, q):
    # cpp: kruells_shockaccel2_diffusion
    diffval = np.sqrt(2.0 * q * (a + b)**2)

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

def nosplit(t, x, last_t, last_x, w):
    return False

def split(t, x, last_t, last_x, w, split_lim):
    #if x[1] > 1e2 and x[1] / last_x[1] >= split_lim - 0.3:
    #    return True
    if x[1] / last_x[1] >= split_lim:#1.41:
        return True
    else:
        return False


cachedir = "cache"
figdir = "figures"

name = "cooling_test_kruells"



parameters = {
         'Xsh' : 0.001215, 
         'a' : 0.04,
         'b' : 0.02,
         'q' : 1, # kappa is const and is q * (a+b)**2 which is 0.0036
     }
#'k_syn' : 0.02778, # following kruells94 (sec. 4.2.)

# parameters from tests.py kruellsB1c
# gamma = 0.2
# a1 = 9
# r = 3
# ksyn = 0.005
# -> beta_s = sqrt(4 * gamma) = 0.894
# -> kappa = gamma / (a1 * ksyn) = q * beta_s**2
#    with q = 1 / (4 * a1 * ksyn) 
#parameters = {
#        'Xsh' : 0.25,
#        'a' : 0.596,
#        'b' : 0.298,
#        'k_syn': 0.005,
#        'q' : 5.556, 
#    }

nfig = NodeFigure(formats.doublehist)

runs = [#(0.0, 1.8, 800.0),
        # (0.02778, 1.8, 800.0),
        (0.02778, 1.8, 400.0),
        # (0.02778, 1.8, 200.0),
        # (0.02778, 1.8, 100.0),
       ]

for k_syn, split_lim, T in runs:
    n_particle = 8000
    t_inj = T / n_particle
    x0 = np.array([0.0, 1.0])
    dt = 0.01
    confine_x=0.1

    init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]
    #init = [(0.0, np.copy(x0)) for i in range(n_particle)]

    sde = sdes.SDE( init, drift, diffusion, boundaries, split)
    sde.set_parameters(parameters | {'k_syn' : k_syn, 'split_lim': split_lim})

    cache = PickleNodeCache(cachedir, name)

    solvernode = SDESolverNode(f'solver_{k_syn}_T={T}', sde=sde, scheme=b'euler', timestep=dt, observation_times=[T], nthreads=64, cache=cache, splitted=True)

    histo_opts = {'bin_count' : 30, 'plot' : True, 'cache' : cache, 'ignore_cache' : False, 'label': f'$ksyn={{sde.parameters[k_syn]}}$'}
    valuesx = SDEValuesNode(f'valuesx_{k_syn}_T={T}', {'x' : solvernode['solution'][T]['x'], 'weights': solvernode['solution'][T]['weights']}, index=0, T=T, cache=cache)
    valuesp = SDEValuesNode(f'valuesp_{k_syn}_T={T}', {'x' : solvernode['solution'][T]['x'], 'weights': solvernode['solution'][T]['weights']}, index=1, T=T, cache=cache,
        confine_range=[(0, -confine_x, confine_x)],
    )
    print(valuesp.data)
    histogramx = HistogramNode(f'histox_{k_syn}_T={T}', {'values' : valuesx['values'], 'weights' : valuesx['weights']}, log_bins=False, normalize='width', **histo_opts)
    histogramp = HistogramNode(f'histop_{k_syn}_T={T}', {'values' : valuesp['values'], 'weights' : valuesp['weights']}, log_bins=True, normalize='density', ingore_cache=True, show_errors=False, transform=lambda p, N : (p, N*p**2), **histo_opts)#
    print(histogramp.data)
    nfig.add(histogramx, 0)
    nfig.add(histogramp, 1)

nfig.savefig(figdir + '/' + name + '.pdf')


#import cProfile
#pr = cProfile.Profile()
#cProfile.run('kruells9a1_newstyle()', filename="test-cpp-propagation.perf")
