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

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#logging.getLogger("grapheval.node").setLevel(logging.DEBUG)


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

def split(t, x, last_t, last_x, w):
    if x[1] / last_x[1] >= 2.5:#1.8:#1.41:
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

common_parameters = {
        #'beta_s' : 0.06,
        'dt' : 0.05,
        'r' : 4,
    }
confine_x_per_dx_adv = 1800
n_timesteps_per_delta_inv = 4000
n_particles = 6000
x0 = np.array([0.0, 1.0])

name = "dxadv_dxdiff"
figdir = "figures"
cache = PickleNodeCache('cache', name)

#oldkw = cache.load_kwargs('solver_dx_adv=0.5-delta_inv=2.4') 
#print("old kwargs: ", oldkw['sde'].parameters)
#oldkw = cache.load_kwargs('solver_dx_adv=0.1-delta_inv=2.4') 
#print("old kwargs: ", oldkw['sde'].parameters)
#exit()

def datapoint(delta_inv, dx_adv, sigma, store_histogram=True):
    this_name = f"dx_adv={dx_adv}-delta_inv={delta_inv}"

    n_timesteps = delta_inv**2 * n_timesteps_per_delta_inv
    parameters_raw, _ = param_from_numerical(dx_adv=dx_adv, delta=1/delta_inv, sigma=sigma, n_timesteps=n_timesteps, **common_parameters)
    parameters = {
                  'k_syn' : 0,#.0001,
                } | parameters_raw 
    dt = parameters_raw['dt']
    T = parameters_raw['Tmax']
    t_inj = T / n_particles
    
    init = [(i * t_inj, np.copy(x0)) for i in range(n_particles)]

    sde = sdes.SDE(2, init, drift, diffusion, split=split)
    sde.set_parameters(parameters)
    print("param:", dx_adv, delta_inv, parameters)

    solvernode = SDESolverNode('solver_' + this_name, sde=sde, scheme=b'semiimplicit_weak', timestep=dt, observation_times=[T], nthreads=8, cache=cache, splitted=True, ignore_cache=False, supervise=True)

    histo_opts = {'bin_count' : 60, 'cache' : cache, 'ignore_cache' : False, 'label': 'T={T}, splitted: {splitted}', 'plot': 'hist'}

    confine_x = confine_x_per_dx_adv * dx_adv
    valuesp = SDEValuesNode('valuesp_' + this_name, {'x' : solvernode[T]['x'], 'weights': solvernode[T]['weights']}, index=1, T=T, cache=cache,
            confine_range=[(0, -confine_x, confine_x)],
        )
    histogramp = HistogramNode('histop_' + this_name, {'values' : valuesp['values'], 'weights' : valuesp['weights']}, log_bins=True, normalize='density', **histo_opts)

    powerlaw = PowerlawNode('pl_' + this_name, {'dataset' : histogramp}, plot='hist', cache=cache)

    datapoint = NodeGroup('datapoint_' + this_name, {'x' : PassiveNode('xval', obj=delta_inv) , 'y': powerlaw[1], 'dy' : powerlaw[3]}, cache=cache)

    if store_histogram:
        valuesx = SDEValuesNode('valuesx_' + this_name, {'x' : solvernode[T]['x'], 'weights': solvernode[T]['weights']}, index=0, T=T, cache=cache)
        histogramx= HistogramNode('histox_' + this_name, {'values' : valuesx['values'], 'weights' : valuesx['weights']}, log_bins=False, normalize='weight', **histo_opts)

        nfig = NodeFigure(formats.doublehist, suptitle=f"deltainv={delta_inv}")
        nfig.add(histogramx, 0, plot_on="hist")
        nfig.add(powerlaw, 1, plot_on="hist")
        nfig.savefig(f"{figdir}/{name}_{this_name}.pdf")
    
    return datapoint

def datarow(delta_invs, dx_adv, sigma):
    pls = {di : datapoint(di, dx_adv, sigma) for di in delta_invs}
    sn = ScatterNode(f'row_dxadv={dx_adv}', parents=pls, plot=True, cache=cache)
    return sn

def rows(delta_invs, dx_advs, sigma):
    allrows = {da : datarow(delta_invs, da, sigma) for da in dx_advs}
    return allrows

#allrows = rows(delta_invs=[1.01, 1.1, 1.2, 1.5, 1.8, 2.4, 3.0, 3.2, 3.4, 3.6], dx_advs=[0.005, 0.05, 0.1, 0.5], sigma=0.95)
allrows = rows(delta_invs=[3.0, 3.2, 3.4], dx_advs=[0.05], sigma=0.95)

xlabel = '\\delta^{-1} = \\Delta x_\\textrm{diff}/\\Delta x_\\textrm{adv}'
nfig = NodeFigure(formats.powerlaws, suptitle='Ratio of diffusive and advective steps', xlabel='$' + xlabel + '$')
for dx_adv, row in allrows.items():
    print("row123", dx_adv, row.data)
    nfig.add(row, 0)
nfig.pad(.2)
#nfig[0].annotate('$\\Delta\\tau =' + str(param['dt']) + '$\n$\\sigma=0.5$', (0.50, 0.18), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
nfig[0].legend(ncols=1, title='$\\Delta x_\\textrm{adv}$')
nfig[0].format(ylim=(-3.5, -1.8))
nfig._legends_kw = {}
print(nfig._chains)
nfig.savefig(f"figures/{name}.pdf")
