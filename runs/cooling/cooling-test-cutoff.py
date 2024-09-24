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

def drift(out, t, x, Xsh, a, b, k_syn, q):
    # cpp: kruells_shockaccel2_drift_94_2

    v0 = kruells94_beta(x[0], Xsh, a, b)
    v1 = - (x[1]) * (kruells94_dbetadx(x[0], Xsh, b) / 3 + k_syn * x[1])

    out[0] = v0
    out[1] = v1

def diffusion(out, t, x, a, b, q):
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

def split(t, x, last_t, last_x, w, split_lim1, split_lim2, cutoff):
    if x[1] > cutoff and x[1] / last_x[1] >= split_lim2:
        return True
    if x[1] / last_x[1] >= split_lim1:#1.41:
        return True
    else:
        return False

def webb84_from_params(params):
    r = (params['a'] + params['b']) / (params['a'] - params['b'])
    a1 = 1 / (4 * params['q'] * params['k_syn'])
    a2 = a1 / r**2

    return r, a1, a2

def cutoff_webb84(r, a1, a2):
    # fig. 3 of webb1984
    return 4 / 3 * (r - 1) / (r / a1 + 1 / a2)

def q_from_cutoff_webb84(cutoff, r, k_syn):
    return 1 / (3 * cutoff * k_syn * r) * (r - 1) / (r + 1)

def k_syn_from_cutoff_webb84(cutoff, r, q):
    return 1 / (3 * cutoff * q * r) * (r - 1) / (r + 1)

cachedir = "cache"
figdir = "figures"

name = "cooling_test_cutoff"

parameters = {
       'Xsh' : 0.25,
       'a' : 0.596,
       'b' : 0.298,
       'q' : 5.556, 
   }

runs = [
        # k_syn, split_lim1, split_lim2, cutoff
        #(0.0, 1.7, np.inf),
        #(1e-6, 1.6, 1.06),
        #(3.33e-6, 1.6, 1.06),
        (1e-5, 1.6, 1.06, 2e5),
       ]

obs_T = np.array([20, 60, 200, 600, 1000, 1200, 1400, 1600, 1800, 2000], dtype=np.float64)
show_T = np.array([600, 1000, 2000], dtype=np.float64)

def do(init, rname):
    nfig = NodeFigure(formats.doublehist2)
    cycle = iter(pplt.Cycle('default'))
    handles = []
    cutoff_exps = []

    for k_syn, split_lim1, split_lim2, particle_lim in runs:
        this_color = next(cycle)['color']
        confine_x = np.inf
        parameters_this = parameters | {'k_syn' : k_syn, 'split_lim1': split_lim1, 'split_lim2': split_lim2}
        if k_syn > 0:
            cutoff_expectation = cutoff_webb84(*webb84_from_params(parameters_this))
            cutoff_exps.append(cutoff_expectation)
        else:
            cutoff_expectation = np.inf


        sde = sdes.SDE(init, drift, diffusion, boundaries, split)
        sde.set_parameters(parameters_this | {'cutoff': cutoff_expectation})

        cache = PickleNodeCache(cachedir, name)

        solvernode = SDESolverNode(f'solver_{rname}_{k_syn}', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_T, nthreads=64, cache=cache, splitted=True, particle_count_limit=particle_lim)

        for T in show_T:
            histo_opts = {'bin_count' : 30, 'plot' : True, 'cache' : cache, 'ignore_cache' : False, 'label': '$k_\\mathrm{{syn}}={sde.parameters[k_syn]}$, $T={T}$'}
            valuesx = SDEValuesNode(f'valuesx_{rname}_{k_syn}_T={T}', {'x' : solvernode['solution'][T]['x'], 'weights': solvernode['solution'][T]['weights']}, index=0, T=T, cache=cache)
            valuesp = SDEValuesNode(f'valuesp_{rname}_{k_syn}_T={T}', {'x' : solvernode['solution'][T]['x'], 'weights': solvernode['solution'][T]['weights']}, index=1, T=T, cache=cache,
                confine_range=[(0, -confine_x, confine_x)],
            )
            if T != max(show_T):
                histo_opts |= {'plot_kwargs' : {'linewidth': 0.3}, 'show_errors': False}
            histogramx = HistogramNode(f'histox_{rname}_{k_syn}_T={T}', {'values' : valuesx['values'], 'weights' : valuesx['weights']}, log_bins=False, normalize='width', **histo_opts)
            histogramp = HistogramNode(f'histop_{rname}_{k_syn}_T={T}', {'values' : valuesp['values'], 'weights' : valuesp['weights']}, log_bins=True, normalize='density', **histo_opts)
            #c = pplt.Colormap(this_color, l=0, s=0, alpha=(1.0, 1.0))(np.log(T)/np.log(max(show_T)))
            c = pplt.Colormap(this_color, l=50, s=20, alpha=(0.7, 1.0))(T/max(show_T))
            histogramx.set_color(c)
            histogramp.set_color(c)
            if T == max(show_T):
                nfig.add(VLineNode("vl", parents=histogramp, callback=lambda *a, **_ : cutoff_expectation, plot=True), 1)
                handles += histogramp.handles
            nfig.add(histogramx, 0)
            nfig.add(histogramp, 1)

    nfig[1].legend(handles=handles, loc='ur', ncols=1)
    nfig[1].format(xlim=(1, max(cutoff_exps) * 10))
    nfig.savefig(f"{figdir}/{name}_{rname}.pdf")

n_particle = 1000
t_inj = max(obs_T) / n_particle
x0 = np.array([0.0, 1.0])
dt = 0.001

do([(i * t_inj, np.copy(x0)) for i in range(n_particle)], "constinj")
do([(0.0, np.copy(x0)) for i in range(n_particle)], "deltainj")
