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

def boundaries(t, x):
    ###
    return 0
    ###

    x_a = carray(x, (2,))
    if np.abs(x_a[0]) > 0.005:
        return 1
    else:
        return 0

def nosplit(t, x, last_t, last_x):
    return False

def split(t, x, last_t, last_x, w):
    #if w < 0.05:
    #    return False
    if x[1] / last_x[1] >= 1.8:#1.41:
        return True
    else:
        return False

def param_from_numerical(dx_adv, delta, sigma, dt, r, n_timesteps):
    """
    from outdated chains.py
    """
    dx_diff = dx_adv / delta

    beta_s = (dx_adv - dx_diff / 4) / dt
    #dt = (dx_adv - dx_diff / 4) / beta_s
    q = dt / (dx_adv / dx_diff - 0.25)**2
    assert q > 0
    assert dt > 0
    Xsh = dx_adv * (1 - sigma) + sigma * dx_diff
    Tmax = n_timesteps * dt

    a = beta_s / 2 * (1 + 1 / r)
    b = beta_s / 2 * (r - 1) / r
    
    param_sim = {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh, 'dt' : dt, 'Tmax' : Tmax, 'r' : r, 'a' : a, 'b' : b}
    param_num = {'dx_adv' : dx_adv, 'dx_diff' : dx_diff, 'delta' : delta, 'sigma' : sigma}
    return param_sim, param_num



cachedir = "cache"
figdir = "figures"

name = "mle-pl"

parameters_raw, _ = param_from_numerical(dx_adv=0.1, delta=1/2.3, sigma=0.45, n_timesteps=10580, dt=0.05, r=4)
parameters = {
              'k_syn' : 0,#.0001,
            } | parameters_raw 
dt = parameters_raw['dt']
T = parameters_raw['Tmax']
n_particle = 1000
t_inj = T / n_particle

x0 = np.array([0.0, 1.0])
confine_x=4

init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]

sde = sdes.SDE(2, init, drift, diffusion, boundaries, split)

sde.set_parameters(parameters)

cache = PickleNodeCache(cachedir, name)

#obs_at = [T/8, T/4, T / 2, T]
solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=[T], nthreads=64, cache=cache, splitted=True)

histo_opts = {'bin_count' : 60, 'plot' : True, 'cache' : cache, 'ignore_cache' : False, 'label': 'T={T}, splitted: {splitted}'}

valuesx = SDEValuesNode(f'valuesx', {'x' : solvernode['solution'][T]['x'], 'weights': solvernode['solution'][T]['weights']}, index=0, T=T, cache=cache)
valuesp = SDEValuesNode(f'valuesp', {'x' : solvernode['solution'][T]['x'], 'weights': solvernode['solution'][T]['weights']}, index=1, T=T, cache=cache,
    confine_range=[(0, -confine_x, confine_x)],
)
histogramx = HistogramNode(f'histox', {'values' : valuesx['values'], 'weights' : valuesx['weights']}, log_bins=False, normalize='width', **histo_opts)
histogramp = HistogramNode(f'histop', {'values' : valuesp['values'], 'weights' : valuesp['weights']}, log_bins=True, normalize='density', **histo_opts)
ccdfp = CCDFNode(f'ccdfp', histogramp, **histo_opts)
mlepl = MLEPowerlawNode('mlepl', {'values' : valuesp['values'], 'weights' : valuesp['weights']}, **(histo_opts | {'ignore_cache': True}))

nfig = NodeFigure(formats.momentumhist)
nfig.add(histogramp, 0)
nfig.add(ccdfp, 0)
nfig.add(mlepl, 0)
nfig.savefig(figdir + '/' + name + '.pdf')
