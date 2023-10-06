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

from numba import njit, f8, carray, i1
import numba

import inspect 
import logging
import ctypes
import warnings

#warnings.simplefilter("error", np.VisibleDeprecationWarning)

from sdesolver.util.cprint import cprint_double_cfunc
@njit(i1(f8))
def cprint(f):
    cprint_double_cfunc(f)
    return 0

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
    vp = - (x[2]) * (kruells94_dbetadx(x[0], Xsh, b) / 3 + k_syn * x[2])

    out[0] = v0
    out[1] = 0
    out[2] = vp

def diffusion(out, t, x, Xsh, a, b, q, kappa_perp):
    # cpp: kruells_shockaccel2_diffusion
    diffval = np.sqrt(2.0 * kruells94_kappa_dep(x[0], Xsh, a, b, q))

    # here carray is required to reshape the contiguous pointer
    out_a = carray(out, (3, 3))
    out_a[0, 0] = diffval
    out_a[1, 0] = 0
    out_a[2, 0] = 0
    out_a[0, 1] = 0
    out_a[1, 1] = kappa_perp
    out_a[2, 1] = 0
    out_a[0, 2] = 0
    out_a[1, 2] = 0
    out_a[2, 2] = 0

def boundaries(t, x):
    ###
    return 0
    ###

    x_a = carray(x, (3,))
    if np.abs(x_a[0]) > 0.005:
        return 1
    else:
        return 0

def nosplit(t, x, last_t, last_x, w):
    return False

def split(t, x, last_t, last_x, w):
    if x[2] / last_x[2] >= 1.8:#1.41:
        return True
    else:
        return False


cachedir = "cache"
figdir = "figures"

name = "2d-spatial-test"


#T = 200.0
#t_inj = 0.05
T = 40.0
t_inj = 0.5
x0 = np.array([0.0, 0.0, 1.0])
dt = 0.001
confine_x=100
n_particle = int(T / t_inj) 

init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]
#init = [(0.0, np.copy(x0)) for i in range(n_particle)]

parameters = {
        'Xsh' : 0.001215, 
        'a' : 0.0375,
        'b' : 0.0225,
        'k_syn' : 0,
        'q' : 1,
        'kappa_perp': 0.01, 
    }

sde = sdes.SDE(2, init, drift, diffusion, boundaries, split)
sde.set_parameters(parameters)

cache = PickleNodeCache(cachedir, name)

obs_at = [T/8, T/4, T / 2, T]
solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=1, cache=cache, splitted=True)

histo_opts = {'bin_count' : 30, 'plot' : True, 'cache' : cache, 'ignore_cache' : False, 'label': 'T={T}'}

valuesx0 = {}
valuesx1 = {}
valuesp = {}
histogramx0 = {}
histogramx1 = {}
histogramp = {}
for T_ in obs_at:
    valuesx0[T_] = SDEValuesNode(f'valuesx0_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=0, T=T_, cache=cache)
    valuesx1[T_] = SDEValuesNode(f'valuesx1_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=1, T=T_, cache=cache)
    valuesp[T_] = SDEValuesNode(f'valuesp_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=2, T=T_, cache=cache,
        confine_range=[(0, -confine_x, confine_x)],
    )
    histogramx0[T_] = HistogramNode(f'histox0_{T_}', {'values' : valuesx0[T_]['values'], 'weights' : valuesx0[T_]['weights']}, log_bins=False, normalize='width', **histo_opts)
    histogramx1[T_] = HistogramNode(f'histox1_{T_}', {'values' : valuesx1[T_]['values'], 'weights' : valuesx1[T_]['weights']}, log_bins=False, normalize='width', **histo_opts)
    histogramp[T_] = HistogramNode(f'histop_{T_}', {'values' : valuesp[T_]['values'], 'weights' : valuesp[T_]['weights']}, log_bins=True, normalize='density', **histo_opts)

histogramx0_group = NodeGroup('histox0_group', histogramx0, cache=cache)
histogramx1_group = NodeGroup('histox1_group', histogramx1, cache=cache)
histogramp_group = NodeGroup('histop_group', histogramp, cache=cache)

import proplot as pplt

### varying k0

histox0s_kappa_perp = {}
histox1s_kappa_perp = {}
histops_kappa_perp = {}
kappa_perps = [0.001, 0.01, 0.1, 1, 10]
for kappa_perp in kappa_perps:
    sde_copy = sdes.SDE(2, init, drift, diffusion, boundaries, split)
    sde_copy.set_parameters(parameters | {'kappa_perp': kappa_perp})
    histox0s_kappa_perp[kappa_perp] = histogramx0_group.copy(f"_kappap={kappa_perp}", last_kwargs={'sde': sde_copy})
    histox1s_kappa_perp[kappa_perp] = histogramx1_group.copy(f"_kappap={kappa_perp}", last_kwargs={'sde': sde_copy})
    histops_kappa_perp[kappa_perp] = histogramp_group.copy(f"_kappap={kappa_perp}", last_kwargs={'sde': sde_copy})

format_kappa_perps = NodeFigureFormat(
        subplots={'ncols' :3, 'nrows': len(kappa_perps), 'sharex': True},
        fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Perpendicular diffusion test'},
        axs_format=[{'xscale': 'linear', 'xformatter': pplt.SciFormatter(), 'xlabel': '$z$'},
                    {'xscale': 'linear', 'xformatter': pplt.SciFormatter(), 'xlabel': '$x$'},
                    {'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$'}] * len(kappa_perps),
        legends_kw=[None, None, {'loc': 'ur', 'ncols': 1}] * len(kappa_perps)
    )

nfig_kappa_perp_double = NodeFigure(formats.doublehist)
for i, kappa_perp in enumerate(kappa_perps):
    kappa_perp = float(kappa_perp)
    nfig_kappa_perp_double.add(histox0s_kappa_perp[kappa_perp][T], 0, label=f"$\kappa_\perp={kappa_perp:.2}$")
    nfig_kappa_perp_double.add(histox1s_kappa_perp[kappa_perp][T], 1, label=f"$\kappa_\perp={kappa_perp:.2}$")
    nfig_kappa_perp_double.add(histops_kappa_perp[kappa_perp][T], 2, label=f"$\kappa_\perp={kappa_perp:.2}$")#, label="test")
nfig_kappa_perp_double.savefig(figdir + '/' + name + '-kappa_perp-triple.pdf')



