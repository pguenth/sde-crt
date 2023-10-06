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

@njit(f8(f8, f8, f8))
def momentum_dep_kappa(p, kappa_0, exp):
    return kappa_0 * p**(2 - exp)

@njit(f8(f8, f8, f8))
def momentum_dep_a2(p, a2_0, exp):
    return a2_0 * p**(exp - 2)

def drift_spatialkappa(out, t, x, Xsh, a, b, k_syn, q):
    # cpp: kruells_shockaccel2_drift_94_2

    v0 = kruells94_beta(x[0], Xsh, a, b)
    v1 = - (x[1]) * (kruells94_dbetadx(x[0], Xsh, b) / 3 + k_syn * x[1])

    out[0] = v0
    out[1] = v1

def drift(out, t, x, a2_0, exp, k_syn):
    # cpp: kruells_2ndorder_drift
    out[0] = 0
    out[1] = x[1] * (4 * momentum_dep_a2(x[1], a2_0, exp) - k_syn * x[1])

def diffusion(out, t, x, kappa_0, a2_0, exp):
    # cpp: kruells_2ndorder_diffusion

    # here carray is required to reshape the contiguous pointer
    out_a = carray(out, (2, 2))
    out_a[0, 0] = np.sqrt(2.0 * momentum_dep_kappa(x[1], kappa_0, exp))
    out_a[1, 0] = 0
    out_a[0, 1] = 0
    out_a[1, 1] = np.sqrt(2.0 * momentum_dep_a2(x[1], a2_0, exp)) * x[1]

def boundaries(t, x, L):#, Lyupper, Lylower):
    x_a = carray(x, (2,))
    if np.abs(x_a[0]) > L:
        return 1
    #elif x_a[1] > Lyupper:
    #    return 2
    #elif x_a[1] < Lylower:
    #    return 3
    else:
        return 0

def nosplit(t, x, last_t, last_x, w):
    return False

def split(t, x, last_t, last_x, w):
    if x[1] / last_x[1] >= 1.8:#1.41:
        return True
    else:
        return False


cachedir = "cache"
figdir = "figures"

name = "momdepkap2"


#T = 200.0
#t_inj = 0.05
T = 80.0
t_inj = 0.005
x0 = np.array([0.0, 1.0])
dt = 0.001
confine_x=40
n_particle = int(T / t_inj) 

init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]
#init = [(0.0, np.copy(x0)) for i in range(n_particle)]

parameters = {
        #'Xsh' : 0.001215, 
        #'a' : 0.0375,
        #'b' : 0.0225,
        'k_syn' : 0,
        'q' : 1,
        'kappa_0': 1, 
        'exp' : 5/3,
        'a2_0' : 0.1,
        'L' : 40

    }

#dummy
sde = sdes.SDE(2, init, drift, diffusion, boundaries, nosplit)
sde.set_parameters(parameters)

cache = PickleNodeCache(cachedir, name)

obs_at = [T]#[T/8, T/4, T / 2, T]
solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=64, cache=cache, splitted=False)

histo_opts = {'bin_count' : 50, 'plot' : True, 'cache' : cache, 'ignore_cache' : False, 'label': 'T={T}'}

valuesx = {}
valuesp = {}
histogramx = {}
histogramp = {}
for T_ in obs_at:
    valuesx[T_] = SDEValuesNode(f'valuesx_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=0, T=T_, cache=cache)
    valuesp[T_] = SDEValuesNode(f'valuesp_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=1, T=T_, cache=cache,
        confine_range=[(0, -confine_x, confine_x)],
    )
    histogramx[T_] = HistogramNode(f'histox_{T_}', {'values' : valuesx[T_]['values'], 'weights' : valuesx[T_]['weights']}, log_bins=False, normalize='width', **histo_opts)
    histogramp[T_] = HistogramNode(f'histop_{T_}', {'values' : valuesp[T_]['values'], 'weights' : valuesp[T_]['weights']}, log_bins=True, normalize='density', transform=lambda p, F : (p, p * F), **histo_opts)

histogramx_group = NodeGroup('histox_group', histogramx, cache=cache)
histogramp_group = NodeGroup('histop_group', histogramp, cache=cache)

import proplot as pplt

### varying a2_0

histoxs_a2_0 = {}
histops_a2_0 = {}
a2_0s = [0.01, 0.1, 1]
hmin = []
hmax = []
for a2_0 in a2_0s:
    sde_copy = sdes.SDE(2, init, drift, diffusion, boundaries, nosplit)
    sde_copy.set_parameters(parameters | {'a2_0': a2_0})
    histoxs_a2_0[a2_0] = histogramx_group.copy(f"_a2_0={a2_0}", last_kwargs={'sde': sde_copy})
    histops_a2_0[a2_0] = histogramp_group.copy(f"_a2_0={a2_0}", last_kwargs={'sde': sde_copy})
    
    # dont know why this is neccessary 
    hp = histops_a2_0[a2_0][T].data[0]
    hmin.append(min(hp))
    hmax.append(max(hp))

nfig_a2_0_double = NodeFigure(formats.doublehist)
for i, a2_0 in enumerate(a2_0s):
    a2_0 = float(a2_0)

    nfig_a2_0_double.add(histoxs_a2_0[a2_0][T], 0, label=f"$a2_0={a2_0:.2}$")
    nfig_a2_0_double.add(histops_a2_0[a2_0][T], 1, label=f"$a2_0={a2_0:.2}$")

nfig_a2_0_double.format(xlim=(min(hmin), max(hmax)))
nfig_a2_0_double.savefig(figdir + '/' + name + '-a2_0-double.pdf')


### varying exp
histoxs_exp = {}
histops_exp = {}
exps = [0, 1, 5/3]
for exp in exps:
    sde_copy = sdes.SDE(2, init, drift, diffusion, boundaries, nosplit)
    sde_copy.set_parameters(parameters | {'exp': exp})
    histoxs_exp[exp] = histogramx_group.copy(f"_exp={exp}", last_kwargs={'sde': sde_copy})
    histops_exp[exp] = histogramp_group.copy(f"_exp={exp}", last_kwargs={'sde': sde_copy})

format_exps = NodeFigureFormat(
        subplots={'ncols' :2, 'nrows': len(exps), 'sharex': True},
        fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': '$\kappa = \kappa_0 p^\epsilon$: Varying $\epsilon$'},
        axs_format=[{'xscale': 'linear', 'xformatter': pplt.SciFormatter(), 'xlabel': '$x$'},
                    {'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$'}] * len(exps),
        legends_kw=[None, {'loc': 'ur', 'ncols': 1}] * len(exps)
    )

#nfig_exp = NodeFigure(format_exps)
#for i, exp in enumerate(exps):
#    nfig_exp.add(histoxs_exp[exp], 2 * i)
#    nfig_exp.add(histops_exp[exp], 2 * i + 1)
#nfig_exp.savefig(figdir + '/' + name + '-exp.pdf')

nfig_exp_double = NodeFigure(formats.doublehist)
for i, exp in enumerate(exps):
    exp = float(exp)
    nfig_exp_double.add(histoxs_exp[exp][T], 0, label=f"$\epsilon={exp:.2}$")
    nfig_exp_double.add(histops_exp[exp][T], 1, label=f"$\epsilon={exp:.2}$")#, label="test")
nfig_exp_double.savefig(figdir + '/' + name + '-exp-double.pdf')
