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

import proplot as pplt

logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def param_from_numerical(dx_adv, delta, sigma, dt, r, Tmax):
    """
    from outdated chains.py
    """
    dx_diff = dx_adv / delta

    beta_s = (dx_adv - dx_diff / 4) / dt
    q = dt / (dx_adv / dx_diff - 0.25)**2
    assert q > 0
    assert beta_s > 0
    Xsh = dx_adv * (1 - sigma) + sigma * dx_diff

    a = beta_s / 2 * (1 + 1 / r)
    b = beta_s / 2 * (r - 1) / r
    
    param_sim = {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh, 'dt' : dt, 'Tmax' : Tmax, 'r' : r, 'a' : a, 'b' : b}
    param_num = {'dx_adv' : dx_adv, 'dx_diff' : dx_diff, 'delta' : delta, 'sigma' : sigma, 'ntimesteps' : Tmax / dt}
    return param_sim, param_num

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

def split(t, x, last_t, last_x, w):
    if x[1] / last_x[1] >= 1.8:
        return True
    else:
        return False

common_parameters = {
        #'beta_s' : 0.06,
        #'dt' : 0.05,
        'r' : 4,
        'sigma' : 0.5,
        'delta' : 0.5
    }
confine_x_per_dx_adv = 40
x0 = np.array([0.0, 1.0])
n_particles = 1000

name = "equilibrium"
figdir = "figures"
cache = PickleNodeCache('cache', name)
def datapoint(dt, Tmax, dx_adv, store_histogram=True):
    this_name = f"Tmax={Tmax}_dt={dt}"

    parameters_raw, parameters_num = param_from_numerical(dt=dt, Tmax=Tmax, dx_adv=dx_adv, **common_parameters)
    parameters = {
                  'k_syn' : 0,#.0001,
                } | parameters_raw 
    t_inj = Tmax / n_particles 
    
    init = [(i * t_inj, np.copy(x0)) for i in range(n_particles)]

    sde = sdes.SDE(2, init, drift, diffusion, split=split)
    sde.set_parameters(parameters)
    #print("param:", dx_adv, delta_inv, parameters)

    solvernode = SDESolverNode('solver_' + this_name, sde=sde, scheme=b'euler', timestep=dt, observation_times=[Tmax], nthreads=8, cache=cache, splitted=True, ignore_cache=False, supervise=True)

    histo_opts = {'bin_count' : 60, 'cache' : cache, 'ignore_cache' : False, 'label': 'Tmax={Tmax}, splitted: {splitted}', 'plot': 'hist'}

    confine_x = confine_x_per_dx_adv * dx_adv
    valuesp = SDEValuesNode('valuesp_' + this_name, {'x' : solvernode[Tmax]['x'], 'weights': solvernode[Tmax]['weights']}, index=1, Tmax=Tmax, cache=cache)
    histogramp = HistogramNode('histop_' + this_name, {'values' : valuesp['values'], 'weights' : valuesp['weights']}, log_bins=True, normalize='density', **histo_opts)

    powerlaw = PowerlawNode('pl_' + this_name, {'dataset' : histogramp}, plot='hist', cache=cache)


    if store_histogram:
        valuesx = SDEValuesNode('valuesx_' + this_name, {'x' : solvernode[Tmax]['x'], 'weights': solvernode[Tmax]['weights']}, index=0, Tmax=Tmax, cache=cache)
        histogramx= HistogramNode('histox_' + this_name, {'values' : valuesx['values'], 'weights' : valuesx['weights']}, log_bins=False, normalize='weight', **histo_opts)

        nfig = NodeFigure(formats.doublehist, suptitle=this_name)
        nfig.add(histogramx, 0, plot_on="hist")
        nfig.add(powerlaw, 1, plot_on="hist")
        nfig.savefig(f"{figdir}/{name}_{this_name}.pdf")
    
    return powerlaw

def datarow_Tmax(Tmaxs, dt, dx_adv):
    pls = {}
    for Tmax in Tmaxs:
        pl = datapoint(dt, Tmax, dx_adv)
        pls[Tmax] = NodeGroup(f'datapoint_{Tmax}', {'x' : PassiveNode('xval', obj=Tmax) , 'y': pl[1], 'dy' : pl[3]})

    sn = ScatterNode(f'row_Tmax', parents=pls, plot=True)
    return sn

def datarow_dt(Tmax, dts, dx_adv_per_dt):
    pls = {}
    for dt in dts:
        pl = datapoint(dt, Tmax, dx_adv_per_dt * dt)
        pls[dt] = NodeGroup(f'datapoint_{dt}', {'x' : PassiveNode('xval', obj=dt) , 'y': pl[1], 'dy' : pl[3]})
    sn = ScatterNode(f'row_dt', parents=pls, plot=True)
    return sn

dtrow = datarow_dt(300.0, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], 0.5)
Tmaxrow = datarow_Tmax([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0], 0.1, 0.5)

dt_label_template = "Varying $\\Delta\\tau$, $\\Delta x_\\mathrm{{adv}}=\\Delta\\tau/2$"
Tmax_label_template = "Varying $T$, $\\Delta x_\\mathrm{{adv}}=1/2$"

#creating one figure

nfig = NodeFigure(formats.powerlaws, xlabel='Runtime $T$', xscale='log', 
                xformatter=pplt.SimpleFormatter(precision=3, prefix="$", suffix="$"))
nfig.add(Tmaxrow)
nfig.format(suptitle='Reaching the temporal equilibrium')
ox = nfig[0].altx()
ox.invert_xaxis()
ox.format(xscale='log', xlabel="Timestep $\\Delta\\tau$")
dtrow(ox, plot_kwargs={'color': 'red'})
nfig[0].legend(handles=dtrow._plot_handles + Tmaxrow._plot_handles, ncols=1)
nfig.pad(.2)
nfig[0].annotate('$\\delta =0.5,~~\\sigma=0.5$', (0.61, 0.3), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
nfig.format(ylim=(-6, -2))
nfig._legends_kw = {}
nfig.savefig("figures/equilibrium_{}.pdf".format(name))
