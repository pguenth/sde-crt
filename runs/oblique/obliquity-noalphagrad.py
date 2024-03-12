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

import proplot as pplt

import formats

from numba import njit, f8, carray, i1
import numba

import inspect 
import logging
import ctypes
import warnings

#warnings.simplefilter("error", np.VisibleDeprecationWarning)
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
   
    param_sim = {'beta_s' : beta_s, 'kappa_par' : q * beta_s**2, 'z_s' : Xsh, 'dt' : dt, 'Tmax' : Tmax, 'r' : r, 'a_beta' : a, 'b_beta' : b}
    param_num = {'dx_adv' : dx_adv, 'dx_diff' : dx_diff, 'delta' : delta, 'sigma' : sigma}
    return param_sim, param_num

from sdesolver.util.cprint import cprint_double_cfunc
@njit(i1(f8))
def cprint(f):
    cprint_double_cfunc(f)
    return 0

logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@njit(f8(f8, f8, f8, f8))
def tanh_profile(x, width, a, b):
    return a - b * np.tanh(x / width)

@njit(f8(f8, f8, f8))
def tanh_profile_diff(x, width, b):
    return - b / (width * np.cosh(x / width)**2)

#@njit(f8(f8, f8, f8, f8, f8))
#def kruells94_kappa_dep(x, Xsh, a, b, q):
#    return q * tanh_profile(x, Xsh, a, b)**2
#
#@njit(f8(f8, f8, f8, f8, f8))
#def kruells94_dkappadx_dep(x, Xsh, a, b, q):
#    return 2 * q * tanh_profile(x, Xsh, a, b) * tanh_profile_diff(x, Xsh, b)

def drift(out, t, x, z_s, a_beta, b_beta, a_alpha, b_alpha, k_syn):
    a = tanh_profile(x[1], z_s, a_alpha, b_alpha)
    beta = tanh_profile(x[1], z_s, a_beta, b_beta)
    dbetadz = tanh_profile_diff(x[1], z_s, b_beta)
    sina = np.sin(a)
    cosa = np.cos(a)

    out[0] = - beta * sina
    out[1] = beta * cosa
    out[2] = - x[2] * (cosa * dbetadz / 3 + k_syn * x[2])

def diffusion(out, t, x, z_s, a_alpha, b_alpha, kappa_perp, kappa_par):
    #a = alpha_u if x < 0 else alpha_d
    a = tanh_profile(x[1], z_s, a_alpha, b_alpha)
    c2a = np.cos(2 * a)
    s2a = np.sin(2 * a)
    ksum = (kappa_par + kappa_perp) / 2
    kdiff = (kappa_par - kappa_perp) / 2
    # z: perp to shock (not neccessarily B), x: parallel to shock
    diff_xx = np.sqrt(2 * (ksum - c2a * kdiff))
    diff_zz = np.sqrt(2 * (ksum + c2a * kdiff))
    diff_anisotrope = np.sqrt(np.abs(- 2 * kdiff * s2a))

    # here carray is required to reshape the contiguous pointer
    out_a = carray(out, (3, 3))
    out_a[0, 0] = diff_xx
    out_a[1, 0] = diff_anisotrope
    out_a[2, 0] = 0
    out_a[0, 1] = diff_anisotrope
    out_a[1, 1] = diff_zz
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

name = "2d-oblique-test-noalphagrad"


parameters = {
        'z_s' : 0.001215, 
        'a_beta' : 0.0375,
        'b_beta' : 0.0225,
        'a_alpha' : 0.0,
        'b_alpha' : 0.0,
        'kappa_perp': 0.0005, 
        'kappa_par': 0.001, 
        'k_syn' : 0,
    }

parameters_new, _ = param_from_numerical(dx_adv=0.1, delta=0.35, sigma=0.95, beta_s=0.06, r=4, n_timesteps=40000) # not working for some reason
#parameters_new, _ = param_from_numerical(dx_adv=0.00053, delta=0.323, sigma=0.95, beta_s=0.06, r=4, n_timesteps=20000) # the "OG" run parameters
#parameters_new, _ = param_from_numerical(dx_adv=0.00053, delta=0.323, sigma=0.95, beta_s=0.99, r=4, n_timesteps=20000) # works qualitatively, powerlaw is steeper
#parameters_new, _ = param_from_numerical(dx_adv=0.00053, delta=0.323, sigma=0.95, beta_s=0.06, r=3, n_timesteps=20000) # works qualitatively, powerlaw is steeper
print(parameters_new, parameters)
parameters |= parameters_new

T = parameters['Tmax']
n_particle = 4000
t_inj = T / n_particle
x0 = np.array([0.0, 0.0, 1.0])
dt = parameters['dt']
confine_x=100

init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]

cache = PickleNodeCache(cachedir, name)

obs_at = [T/8, T/4, T / 2, T]
histo_opts = {'bin_count' : 30, 'plot' : True, 'cache' : cache, 'ignore_cache' : False, 'label': 'T={T}'}


def one_row(rname, param):
    sde = sdes.SDE(init, drift, diffusion, boundaries, split)
    sde.set_parameters(param)
    solvernode = SDESolverNode(f'solver_{rname}', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=64, cache=cache, splitted=True)
    valuesx0 = {}
    valuesx1 = {}
    valuesp = {}
    histogramx0 = {}
    histogramx1 = {}
    histogramp = {}
    for T_ in obs_at:
        valuesx0[T_] = SDEValuesNode(f'valuesx0_{rname}_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=0, T=T_, cache=cache)
        valuesx1[T_] = SDEValuesNode(f'valuesx1_{rname}_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=1, T=T_, cache=cache)
        valuesp[T_] = SDEValuesNode(f'valuesp_{rname}_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=2, T=T_, cache=cache,
                                    #confine_range=[(0, -confine_x, confine_x)],
        )
        histogramx0[T_] = HistogramNode(f'histox0_{rname}_{T_}', {'values' : valuesx0[T_]['values'], 'weights' : valuesx0[T_]['weights']}, log_bins=False, normalize='width', **histo_opts)
        histogramx1[T_] = HistogramNode(f'histox1_{rname}_{T_}', {'values' : valuesx1[T_]['values'], 'weights' : valuesx1[T_]['weights']}, log_bins=False, normalize='width', **(histo_opts | {'plot' : 'hist'}))
        histogramp[T_] = HistogramNode(f'histop_{rname}_{T_}', {'values' : valuesp[T_]['values'], 'weights' : valuesp[T_]['weights']}, log_bins=True, normalize='density', **histo_opts)

    histogramx0_group = NodeGroup(f'histox0_group_{rname}', histogramx0, cache=cache)
    histogramx1_group = NodeGroup(f'histox1_group_{rname}', histogramx1, cache=cache)
    histogramp_group = NodeGroup(f'histop_group_{rname}', histogramp, cache=cache)

    powerlaw = MLEPowerlawNode('mlepl_' + rname, {'values' : valuesp[T]['values'], 'weights' : valuesp[T]['weights']}, plot='hist', cache=cache, ignore_cache=False)

    return histogramx0_group, histogramx1_group, histogramp_group, valuesp[T], powerlaw

histogramx0_group, histogramx1_group, histogramp_group, _, powerlaw = one_row("", parameters)
nfig = NodeFigure(formats.triplehist)
nfig.add(histogramx0_group, 0)
nfig.add(histogramx1_group, 1, plot_on='hist')
nfig.add(histogramp_group, 2)
nfig.add(powerlaw, 2, plot_on='hist')

nfig.savefig(f"{figdir}/{name}-single.pdf")
exit()

### varying upstream angle

histox0groups = {}
histox1groups = {}
histopgroups = {}
alpha_u = np.array([0.0, np.pi / 10, 3 * np.pi / 10, np.pi / 2])
alpha_d = 0.0
multihist = NodeFigureFormat(
        subplots={'ncols' :3, 'nrows': len(alpha_u), 'sharex': False},
        fig_format={'yscale': 'log', 'yformatter': 'log'},
        axs_format=[{'xscale': 'linear', 'xformatter': pplt.SciFormatter(), 'xlabel': '$x$ (parallel to shock)'},
                    {'xscale': 'linear', 'xformatter': pplt.SciFormatter(), 'xlabel': '$z$ (perpendicular to shock)'},
                    {'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$'}] * len(alpha_u),
        legends_kw=[None, None, {'loc': 'ur', 'ncols': 1}]
    )
nfig_multi = NodeFigure(multihist)
for i, a_u in enumerate(alpha_u):
    p = parameters | {
            'a_alpha' : (a_u + alpha_d) / 2,
            'b_alpha' : (a_u - alpha_d) / 2,
            }
    histogramx0_group, histogramx1_group, histogramp_group, v, _ = one_row(f"alpha_u={a_u}", p)
    print(v.data)
    nfig_multi.add(histogramx0_group, i * 3 + 0)
    nfig_multi.add(histogramx1_group, i * 3 + 1, plot_on='hist')
    nfig_multi.add(histogramp_group, i * 3 + 2)

nfig_multi.savefig(f"{figdir}/{name}-multi.pdf")
