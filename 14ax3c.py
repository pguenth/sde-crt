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

from astropy import units as u
from astropy import constants

from toptygin import *

logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from src.radiativenodes import *

#import resource
#s, h = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (s * 10, h * 10))

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

def split(t, x, last_t, last_x, w):
    if x[1] / last_x[1] >= 2.2:
        return True
    else:
        return False

cachedir = "cache"
figdir = "figures"

name = "14ax3c-new"

###
### SDE definition
###

T = 200.0
obs_at = [2.0, 6.4, 20.0, 200.0]
t_inj = 0.01
x0 = np.array([0.0, 1.0])
dt = 0.001
confine_x=1#0.05
n_particle = int(T / t_inj)

init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]

parameters_raw, _ = param_from_numerical(dx_adv=0.00053, delta=0.323, sigma=0.95, beta_s=0.06, r=4, n_timesteps=20000)
parameters = {
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
            } | parameters_raw 
dt = parameters_raw['dt']

sde = sdes.SDE(2, init, drift, diffusion, boundaries, split=split)
sde.set_parameters(parameters)

###
### Solving
###

cache = PickleNodeCache(cachedir, name)
nthreads = 64

solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=nthreads, cache=cache, splitted=True, ignore_cache=False, supervise=False)
#solvernode = SDESolverNode('solver_kppc', sde=sde, scheme=b'kppc', timestep=dt, observation_times=obs_at, nthreads=nthreads, cache=cache, splitted=True, ignore_cache=False, supervise=True)
#solvernode = SDESolverNode('solver_kppc_split', sde=sde, scheme=b'kppc', timestep=dt, observation_times=obs_at, nthreads=nthreads, cache=cache, splitted=True, ignore_cache=False, supervise=True)
#solvernode = SDESolverNode('solver_nosplit', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=4, cache=cache, splitted=False, ignore_cache=False, supervise=True)

histo_opts = {'bin_count' : 60, 'cache' : cache, 'ignore_cache' : True, 'label': 'T={T}, splitted: {splitted}', 'plot': 'hist'}

valuesx = {}
valuesp = {}
histogramx = {}
histogramp = {}
for T_ in obs_at:
    valuesx[T_] = SDEValuesNode(f'valuesx_{T_}', {'x' : solvernode[T_]['x'], 'weights': solvernode[T_]['weights']}, index=0, T=T_, cache=cache, ignore_cache=True)
    valuesp[T_] = SDEValuesNode(f'valuesp_{T_}', {'x' : solvernode[T_]['x'], 'weights': solvernode[T_]['weights']}, index=1, T=T_, cache=cache,
        confine_range=[(0, -confine_x, confine_x)], ignore_cache=True
    )
    histogramx[T_] = HistogramNode(f'histox_{T_}', {'values' : valuesx[T_]['values'], 'weights' : valuesx[T_]['weights']}, log_bins=False, normalize='width', **histo_opts)
    histogramp[T_] = HistogramNode(f'histop_{T_}', {'values' : valuesp[T_]['values'], 'weights' : valuesp[T_]['weights']}, log_bins=True, normalize='density', **histo_opts)

powerlaw = PowerlawNode('pl', {'dataset' : histogramp[T]}, plot=True, plot_kwargs={'color': 'k'})
groupp = NodeGroup('gp', parents=histogramp)
groupx = NodeGroup('gx', parents=histogramx)

###
### normal histogram figure
###

nfig = NodeFigure(formats.doublehist2)
nfig.format(suptitle="Spatial and momentum spectrum of particles for pure diffusive shock acceleration")
nfig[0].format(xlim=(-0.2, 0.6), ylim=(1e-2, 300))
nfig[0].annotate('$\\delta =0.323,~~\\sigma=0.95$', (0.06, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
nfig[1].format(xlim=(1, 3000), ylim=(3e-7, 10), xlabel="$p/p_\\mathrm{inj}$ at the shock")
nfig[1].format(xformatter='log')
nfig.add(groupx, 0)
nfig.add(groupp, 1)
nfig.add(powerlaw, 1)
#handles_ts = []
#for _, n in groupp.parents_iter:
#    handles_ts += n.handles
#nfig[1].legend(loc='ur', handles=handles_ts, ncols=1, title='Observation time $T$')
#nfig[1].legend(loc='ll', handles=powerlaw.handles,ncols=1, title='Powerlaw fit')
nfig.savefig(figdir + '/' + name + '.pdf')

###
### contour figure
###

N0_base = 0.0054 * 200
dict2d = {
    # T : xmin, xmax, pmin, pmax, N0_corr, cut_topt_levels
    #0.64 : (((-0.1, 0.05), (1, 3)), ),
    2.0 : (((-0.15, 0.1), (1, 10)), 1, 5),
    6.4 : (((-0.2, 0.2), (1, 100)), 1, 2),
    20 : (((-0.2, 0.5), (1, 300)), 1, 1),
    200 : (((-0.2, 3.5), (1, 300)), 1, None),
}

topt_param = {
    'x0': 0,
    'y0': 1, 
    'beta_s': parameters['beta_s'],
    'q': parameters['q'],
    'r': parameters['r']
}

contourlevels = np.linspace(-5, 1, 16)
nfigc = NodeFigure(formats.contours4)

from matplotlib import lines

for n, (T, (lims, N0_corr, cut_levels_topt)) in enumerate(dict2d.items()):
    labels_hist=False
    topt_detail=200
    topt_labels=False
    bins=25
    N0 = N0_base / T * N0_corr

    histo2d = Histogram2DNode('histo2d_T=' + str(T), {'values' : solvernode[T]['x'], 'weights' : solvernode[T]['weights']}, bin_count=bins, log_bins=(False, True), plot=True, 
                              style='contour', normalize='density', log_histogram=True, ignore_cache=False, cache=cache, label='fixme',#${T:.1f}$',
                              cmap='Haline', plot_kwargs=dict(labels=labels_hist, levels=contourlevels, cmap_kw={'reverse': True}, robust=True,
                              labels_kw={'color' : 'gray'}), limits=lims)


    toptygin = ToptyginContourNode('topt_T=' + str(T), plot=True, cache=cache,
                    params=topt_param, N0=N0, T=T,
                    x_range=lims[0], y_range=(np.log10(lims[1][0]), np.log10(lims[1][1])),
                    levels=contourlevels, detail=topt_detail,
                    contour_opts=dict(color='k', linestyle='--', label="Analytical solution", linewidth=0.5, alpha=1, labels=topt_labels))
    nfigc[n].format(xlim=lims[0], ylim=lims[1])
    nfigc.add(histo2d, n, instant=False)
    nfigc.add(toptygin, n, instant=False)
    nfigc[n].format(title="Contour plot of $\\log{{\\bar F}}$ at $T={}$".format(T))
    nfigc[n].annotate("$T={}$".format(T), (0.06, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
    if n in [1, 3]:
        nfigc[n].format(ylabel="")
    if n in [0, 1]:
        nfigc[n].format(xlabel="")

nfigc_kppc = nfigc.copy("kppc", last_kwargs=dict(sde=sde, scheme=b'kppc'))
nfigc_semi = nfigc.copy("semiimplicit", last_kwargs=dict(sde=sde, scheme=b'semiimplicit_weak'))
nfigc.execute()

handles_contour = [lines.Line2D([], [], color='k', label='Number density')] + toptygin.handles
hist2dhandle = histo2d.handles
nfigc.figure.colorbar(hist2dhandle, loc='b')
nfigc.figure.legend(loc='t', handles=handles_contour, ncols=2, title="Contour plots of $\\log{{\\bar F}}$")


nfigc.savefig(figdir + '/' + name + '-euler_contours.pdf')
#nfigc_kppc.savefig(figdir + '/' + name + '-kppc_contours.pdf')
nfigc_semi.savefig(figdir + '/' + name + '-semi_contours.pdf')
