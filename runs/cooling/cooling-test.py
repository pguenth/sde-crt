import sdesolver as sdes

import inspect
import time

import proplot as pplt

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

name = "cooling_test"



# parameters = {
#         'Xsh' : 0.001215, 
#         'a' : 0.0375,
#         'b' : 0.0225,
#         'k_syn' : 0,
#         'q' : 1,
#     }
parameters = {
       'Xsh' : 0.25,
       'a' : 0.596,
       'b' : 0.298,
       'q' : 5.556, 
   }


runs = [#(0.0, 1.8, 800.0),
        #(0.00001, 1.55, 800.0),
        #(0.00005, 1.7, 800.0),
        #(0.00008, 1.65, 800.0),
        #(0.0001, 1.6, 800.0),
        #(0.0002, 1.5, 800.0),
        #(0.0005, 1.65, 800.0),
        #(0.001, 1.6, 800.0),
        (0.005, 1.6, 800.0),
        #(0.01, 1.5, 800.0)
       ]

obs_T = np.array([2, 4, 8, 20, 40, 80, 200, 400], dtype=np.float64)

def do(init, rname):
    nfig = NodeFigure(formats.doublehist)

    for k_syn, split_lim, _ in runs:
        confine_x=100

        sde = sdes.SDE( init, drift, diffusion, boundaries, split)
        sde.set_parameters(parameters | {'k_syn' : k_syn, 'split_lim': split_lim})

        cache = PickleNodeCache(cachedir, name)

        solvernode = SDESolverNode(f'solver_{rname}_{k_syn}', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_T, nthreads=64, cache=cache, splitted=True)

        histo_opts = {'bin_count' : 30, 'plot' : True, 'cache' : cache, 'ignore_cache' : False, 'label': '$k_\\mathrm{{syn}}={sde.parameters[k_syn]}$, $T={T}$'}
        for T in obs_T:
            valuesx = SDEValuesNode(f'valuesx_{rname}_{k_syn}_T={T}', {'x' : solvernode['solution'][T]['x'], 'weights': solvernode['solution'][T]['weights']}, index=0, T=T, cache=cache)
            valuesp = SDEValuesNode(f'valuesp_{rname}_{k_syn}_T={T}', {'x' : solvernode['solution'][T]['x'], 'weights': solvernode['solution'][T]['weights']}, index=1, T=T, cache=cache,
                confine_range=[(0, -confine_x, confine_x)],
            )
            histogramx = HistogramNode(f'histox_{rname}_{k_syn}_T={T}', {'values' : valuesx['values'], 'weights' : valuesx['weights']}, log_bins=False, normalize='width', **histo_opts)
            histogramp = HistogramNode(f'histop_{rname}_{k_syn}_T={T}', {'values' : valuesp['values'], 'weights' : valuesp['weights']}, log_bins=True, normalize='density', **histo_opts)
            c = pplt.Colormap('haline')(1 - np.log(T)/np.log(max(obs_T)))
            histogramx.set_color(c)
            histogramp.set_color(c)
            nfig.add(histogramx, 0)
            nfig.add(histogramp, 1)

    nfig.savefig(f"{figdir}/{name}_{rname}.pdf")

n_particle = 20000
t_inj = max(obs_T) / n_particle
x0 = np.array([0.0, 1.0])
dt = 0.001

do([(i * t_inj, np.copy(x0)) for i in range(n_particle)], "constinj")
do([(0.0, np.copy(x0)) for i in range(n_particle)], "deltainj")
