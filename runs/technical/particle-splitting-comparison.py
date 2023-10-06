import sdesolver as sdes
from sdesolver.util.cprint import cprint_double_cfunc

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

import proplot as pplt

logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def param_from_numerical(dx_adv, delta, sigma, beta_s, r, n_timesteps):
    """
    from outdated chains.py
    """
    dx_diff = dx_adv / delta

    dt = (dx_adv - dx_diff / 4) / beta_s
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

@njit(i1(f8))
def cprint(f):
    cprint_double_cfunc(f)
    return 0

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
    x_a = carray(x, (2,))
    if np.abs(x_a[0]) > 0.005:
        return 1
    else:
        return 0

def split(t, x, last_t, last_x, w, split_ratio, min_weight):
    if w < min_weight:
        return False

    if x[1] / last_x[1] >= split_ratio:
        return True
    else:
        return False

class PrintRuntime(EvalNode):
    def plot(self, v, ax, common, **kwargs):
        return ax.annotate(xy=(0.05, 0.05),
                           xycoords='axes fraction',
                           text=f"Runtimes\nsplitted: {v['splitted']:3g}s\nunsplitted: {v['unsplitted']:3g}s",
                           bbox=dict(boxstyle="square,pad=0.3",
                                     fc="white", 
                                     ec="black"
                                     )
                           )



cachedir = "cache"
figdir = "figures"

name = "splitting-comparison"
cache = PickleNodeCache(cachedir, name)

# set the parameters

T = 200.0
obs_at = [T/32, T/16, T / 4, T]
t_inj = 0.05
x0 = np.array([0.0, 1.0])
confine_x=1#0.05
n_particle = int(T / t_inj)

init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]

parameters_raw, _ = param_from_numerical(dx_adv=0.00053, delta=0.323, sigma=0.95, beta_s=0.06, r=4, n_timesteps=10000)
parameters = { 'k_syn' : 0 } | parameters_raw 
parameters_euler = parameters | {'split_ratio': 1.6, 'min_weight': 0.0009} 
parameters_semiimplicit = parameters | {'split_ratio': 3.0, 'min_weight': 0.0009} 

# define the 4 sdes
sde_euler = sdes.SDE(2, init, drift, diffusion, split=split)
sde_euler.set_parameters(parameters_euler)

sde_semiimplicit = sdes.SDE(2, init, drift, diffusion, split=split)
sde_semiimplicit.set_parameters(parameters_semiimplicit)

sde_euler_nosplit = sdes.SDE(2, init, drift, diffusion)
sde_euler_nosplit.set_parameters(parameters_euler)

sde_semiimplicit_nosplit = sdes.SDE(2, init, drift, diffusion)
sde_semiimplicit_nosplit.set_parameters(parameters_semiimplicit)

# create chain for one of the sdes

solvernode = SDESolverNode('solver', sde=sde_semiimplicit, scheme=b'semiimplicit_weak', timestep=parameters['dt'], observation_times=obs_at, nthreads=8, cache=cache, splitted='with splitting', ignore_cache=False, supervise=True)

histo_opts = {'bin_count' : 60, 'cache' : cache, 'ignore_cache' : False, 'plot': True, 'hide_zeros' : False}

valuesx = {}
valuesp = {}
histogramx = {}
histogramp = {}
for T_ in obs_at:
    valuesx[T_] = SDEValuesNode(f'valuesx_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=0, T=T_, cache=cache)
    valuesp[T_] = SDEValuesNode(f'valuesp_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=1, T=T_, cache=cache,
        confine_range=[(0, -confine_x, confine_x)]
    )
    histogramx[T_] = HistogramNode(f'histox_{T_}', {'values' : valuesx[T_]['values'], 'weights' : valuesx[T_]['weights']}, log_bins=False, normalize='width', **histo_opts)
    histogramp[T_] = HistogramNode(f'histop_{T_}', {'values' : valuesp[T_]['values'], 'weights' : valuesp[T_]['weights']}, log_bins=True, normalize='density', label='{splitted}', **histo_opts)

maxpl = PowerlawNode('pl', {'dataset' : histogramp[T]}, plot=True, powerlaw_annotate=False)
groupp = NodeGroup('gp', parents=histogramp)
groupx = NodeGroup('gx', parents=histogramx)

nfig_semiimplicit_small = NodeFigure(formats.doublehist)
nfig_semiimplicit_small.add(maxpl, 1)
nfig_semiimplicit_small.add(histogramx[T], 0)
maxpl_nosplit = maxpl.copy("nosplit", last_kwargs=dict(sde=sde_semiimplicit_nosplit, splitted='without splitting'))
maxhistogramx_nosplit = histogramx[T].copy("nosplit", last_kwargs=dict(sde=sde_semiimplicit_nosplit, splitted='without splitting'))
runtimenode = PrintRuntime("r", parents={'splitted': maxpl.search_parent('solver')['runtime'],
                                         'unsplitted': maxpl_nosplit.search_parent('solver')['runtime']}, plot=True)
nfig_semiimplicit_small.add(maxpl_nosplit, 1)
nfig_semiimplicit_small.add(maxhistogramx_nosplit, 0)
nfig_semiimplicit_small.add(runtimenode, 1)

nfig_semiimplicit_small.savefig(figdir + '/' + name + '-small-semiimplicit.pdf')


maxpl_euler = maxpl.copy("euler", last_kwargs=dict(sde=sde_euler, scheme=b'euler'))
maxpl_euler_nosplit = maxpl_nosplit.copy("euler", last_kwargs=dict(sde=sde_euler_nosplit, scheme=b'euler'))
maxhistogramx_euler = histogramx[T].copy("euler", last_kwargs=dict(sde=sde_euler, scheme=b'euler'))
maxhistogramx_euler_nosplit = maxhistogramx_nosplit.copy("euler", last_kwargs=dict(sde=sde_euler_nosplit, scheme=b'euler'))
nfig_euler_small = NodeFigure(formats.doublehist)
nfig_euler_small.add(maxpl_euler, 1)
nfig_euler_small.add(maxhistogramx_euler, 0)
runtimenode_euler = PrintRuntime("r", parents={'splitted': maxpl_euler.search_parent('solver')['runtime'],
                                         'unsplitted': maxpl_euler_nosplit.search_parent('solver')['runtime']}, plot=True)
nfig_euler_small.add(maxpl_euler_nosplit, 1)
nfig_euler_small.add(maxhistogramx_euler_nosplit, 0)
nfig_euler_small.add(runtimenode_euler, 1)

nfig_euler_small.savefig(figdir + '/' + name + '-small-euler.pdf')

# plot full figure

format8 = NodeFigureFormat(
                subplots={'array': [[1, 2, 5, 6], [3, 4, 7, 8]]},
                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Comparison of runs with and without split particles'},
                axs_format=[{'xscale': 'linear', 'xformatter': pplt.SciFormatter(), 'xlabel': '$x$'}] * 4 + [{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$'}] * 4,
                legends_kw=[{'loc': 'ur', 'ncols': 1}] * 8 
        )

nfig_semiimplicit = NodeFigure(format8)
for i, (_, h) in enumerate(groupx.parents_iter):
    nfig_semiimplicit.add(h, i, instant=False)
for i, (_, h) in enumerate(groupp.parents_iter):
    nfig_semiimplicit.add(h, i + 4, instant=False)

nfig_semiimplicit.add(maxpl, 7, instant=False)

nfig_semiimplicit_nosplit = nfig_semiimplicit.copy("nosplit", last_kwargs=dict(sde=sde_semiimplicit_nosplit, splitted='without splitting'))

# figure with both
nfigboth_semiimplicit = NodeFigure.merge([nfig_semiimplicit, nfig_semiimplicit_nosplit])
nfigboth_semiimplicit.add(runtimenode, 7)

nfig_euler = nfig_semiimplicit.copy("euler", last_kwargs=dict(sde=sde_euler, scheme=b'euler'))
nfig_euler_nosplit = nfig_semiimplicit_nosplit.copy("euler", last_kwargs=dict(sde=sde_euler_nosplit, scheme=b'euler'))
nfigboth_euler = NodeFigure.merge([nfig_euler, nfig_euler_nosplit])
runtimenode_euler = PrintRuntime("r", parents={'splitted': list(nfig_euler._chains.keys())[0].search_parent('solver')['runtime'],
                                         'unsplitted': list(nfig_euler_nosplit._chains.keys())[0].search_parent('solver')['runtime']}, plot=True)
nfigboth_euler.add(runtimenode_euler, 7)

nfigboth_semiimplicit.savefig(figdir + '/' + name + '-both-semiimplicit.pdf')
nfigboth_euler.savefig(figdir + '/' + name + '-both-euler.pdf')
nfig_semiimplicit.savefig(figdir + '/' + name + '-semiimplicit.pdf')
nfig_semiimplicit_nosplit.savefig(figdir + '/' + name + '-semiimplicit-nosplits.pdf')
nfig_euler.savefig(figdir + '/' + name + '-euler.pdf')
nfig_euler_nosplit.savefig(figdir + '/' + name + '-euler-nosplits.pdf')
