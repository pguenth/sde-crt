import sdesolver as sdes
import logging
logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import inspect
import time

from grapheval.nodefigure import NodeFigure, NodeFigureFormat
from grapheval.cache import PickleNodeCache
from grapheval.node import *
from grapheval.graph import draw_node_chain

from src.basicnodes import *
from src.newnodes import *
from src.radiativenodes import *

from scipy import integrate
from scipy.optimize import root_scalar

import formats

from numba import njit, f8, carray
import numba

import inspect 
import ctypes
import warnings

from astropy import units as u
from astropy import constants


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
    if x[1] / last_x[1] >= 1.8:#1.41:
        return True
    else:
        return False

# gauss/tesla equivalency
gauss_tesla_eq = (u.G, u.T, lambda x: x / np.sqrt(4 * np.pi / constants.mu0), lambda x: x * np.sqrt(4 * np.pi / constants.mu0))

def k_syn_from_B(B):
    B_si = B.to("T", equivalencies=[gauss_tesla_eq])
    k_syn = B**2 / (6 * np.pi * constants.eps0) * (constants.e.si / (constants.m_e * constants.c))**4
    return (k_syn / u.Unit("kg-1 m-1")).decompose().value

def B_from_k_syn(k_syn):
    k_syn = k_syn * u.Unit("kg-1 m-1")
    B = (constants.m_e * constants.c / constants.e.si)**2 * np.sqrt(6 * np.pi * constants.eps0 * k_syn)
    return B.to("G", equivalencies=[gauss_tesla_eq])

cachedir = "cache"
figdir = "figures"

name = "naima-test"

T = 200.0
#t_inj = 0.05
#T = 24.5
t_inj = 0.05
x0 = np.array([0.0, 1.0])
dt = 0.001
confine_x=100#0.05
n_particle = int(T / t_inj)

init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]

B = 1 * u.mG
k_syn = k_syn_from_B(B)
print(k_syn)

parameters_raw, _ = param_from_numerical(dx_adv=0.00053, delta=0.323, sigma=0.95, beta_s=0.06, r=4, n_timesteps=20000)
parameters = {
              'k_syn' : 0.0, #k_syn,
              'x0' : 0,
              'y0' : 1,
            } | parameters_raw 
dt = parameters_raw['dt']

sde = sdes.SDE(2, init, drift, diffusion, boundaries, split)
sde.set_parameters(parameters)

# duplicate for the unsplitted variant
#sde_nosplit = sdes.SDE(2, init, drift, diffusion, boundaries)
#sde_nosplit.set_parameters(parameters)


cache = PickleNodeCache(cachedir, name)

obs_at = [T/32, T/16, T / 4, T]
#obs_at = [22.0, 22.5, 23.0, T]
solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=64, cache=cache, splitted=True, ignore_cache=False, supervise=True)

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

maxpl = PowerlawNode('pl', {'dataset' : histogramp[T]}, plot=True)
groupp = NodeGroup('gp', parents=histogramp)
groupx = NodeGroup('gx', parents=histogramx)

energy_range=np.logspace(-5, 14, 100) * u.eV
edist = HistogramElectronEnergyDistribution('edist', parents={'histogram': histogramp[50.0]}, p_inj = 1e1 * constants.m_e * constants.c, n0=1e52)
ic_cmb_naima = ICNaima('iccmbnaima', parents={'electron_distribution': edist}, plot=True, energy_range=energy_range, target=['CMB'], label='IC-CMB')
ic_fir_naima = ICNaima('icfirnaima', parents={'electron_distribution': edist}, plot=True, energy_range=energy_range, target=['FIR'], label='IC-FIR')
ic_nir_naima = ICNaima('icnirnaima', parents={'electron_distribution': edist}, plot=True, energy_range=energy_range, target=['NIR'], label='IC-NIR')
synnaima = SynchrotronNaima('synnaima', parents={'electron_distribution': edist}, plot=True, energy_range=energy_range, B=1*u.mG)
sscnaima = SSCNaima('sscnaima', parents={'electron_distribution': edist}, plot=True, energy_range=energy_range, B=1*u.G, source_size=10 * u.pc)
sednaima = SEDSum('sedsum', parents=[ic_cmb_naima, ic_fir_naima, ic_nir_naima, synnaima, sscnaima], plot=True, plot_kwargs={'linestyle': 'dotted', 'color': 'k'})
radiationgroup = NodeGroup('radgroup', parents=[sednaima, ic_cmb_naima, ic_fir_naima, ic_nir_naima, synnaima, sscnaima])


import proplot as pplt
format9 = NodeFigureFormat(
                subplots={'array': [[1, 2, 5, 6, 9, 9], [3, 4, 7, 8, 9, 9]]},
                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Comparison of runs with and without split particles'},
                axs_format=[{'xscale': 'linear', 'xformatter': pplt.SciFormatter(), 'xlabel': '$x$'}] * 4 
                    + [{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$'}] * 4
                    + [{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': 'Energy in eV', 'ylabel': 'Flux in erg/s'}],
                legends_kw=[{'loc': 'ur', 'ncols': 1}] * 9 
        )
format1 = NodeFigureFormat(
                subplots={'array': [[1]]},
                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Radiation test with naima'},
                axs_format=[{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': 'Energy in eV', 'ylabel': 'Flux $\propto$ erg/s'}],
                legends_kw=[{'loc': 'r', 'ncols': 1}]
        )

nfig_rad = NodeFigure(format1)
nfig_rad.add(radiationgroup, 0, instant=False)
nfig_rad[0].format(ylim=(1e-18, 1e-9))
nfig_rad.savefig(figdir + '/' + name + '-radiationonly.pdf')

nfig = NodeFigure(format9)
for i, (_, h) in enumerate(groupx.parents_iter):
    nfig.add(h, i, instant=False, plot_on='hist')
for i, (_, h) in enumerate(groupp.parents_iter):
    nfig.add(h, i + 4, instant=False, plot_on='hist')

nfig.add(maxpl, 7, instant=False)
nfig.add(radiationgroup, 8, instant=False)

# without splits
nfig2 = nfig.copy("nosplit", last_kwargs=dict(sde=sde_nosplit, splitted=False))

# figure with both
nfigboth = NodeFigure.merge([nfig, nfig2])
nfigboth[7].format(xlim=(1, 1e7))

nfigboth.savefig(figdir + '/' + name + '-both.pdf')
nfig.savefig(figdir + '/' + name + '.pdf')
nfig2.savefig(figdir + '/' + name + '-nosplits.pdf')

#import cProfile
#pr = cProfile.Profile()
#cProfile.run('kruells9a1_newstyle()', filename="test-cpp-propagation.perf")
