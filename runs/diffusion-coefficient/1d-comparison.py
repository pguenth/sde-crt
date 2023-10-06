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

def drift(out, t, x, Xsh, a, b, k_syn, q):
    # cpp: kruells_shockaccel2_drift_94_2

    v0 = kruells94_beta(x[0], Xsh, a, b)
    v1 = - (x[1]) * (kruells94_dbetadx(x[0], Xsh, b) / 3 + k_syn * x[1])

    out[0] = v0
    out[1] = v1

def diffusion(out, t, x, kappa):
    # cpp: kruells_shockaccel2_diffusion
    diffval = np.sqrt(2.0 * kappa)

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

def split(t, x, last_t, last_x, w):
    if x[1] / last_x[1] >= 1.8:#1.41:
        return True
    else:
        return False


cachedir = "cache"
figdir = "figures"

name = "kappa-1dcomp"


#T = 200.0
#t_inj = 0.05
T = 40.0
t_inj = 0.05
x0 = np.array([0.0, 1.0])
dt = 0.001
confine_x=100
n_particle = int(T / t_inj) 

init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]
#init = [(0.0, np.copy(x0)) for i in range(n_particle)]

parameters = {
        'Xsh' : 0.001215, 
        'a' : 0.0,#375,
        'b' : 0.0,#225,
        'k_syn' : 0,
        'q' : 1,
        'kappa': 0.01, 
    }

sde = sdes.SDE(2, init, drift, diffusion, boundaries, split)
sde.set_parameters(parameters)

cache = PickleNodeCache(cachedir, name)

solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=[T], nthreads=64, cache=cache, splitted=True)

histo_opts = {'bin_count' : 30, 'plot' : 'unscaled', 'cache' : cache, 'ignore_cache' : False, 'label': 'T={T}'}

valuesx = {}
valuesx = SDEValuesNode(f'valuesx', {'x' : solvernode['solution'][T]['x'], 'weights': solvernode['solution'][T]['weights']}, index=0, T=T, cache=cache)
histogramx = HistogramNode(f'histox', {'values' : valuesx['values'], 'weights' : valuesx['weights']}, log_bins=False, normalize='width', **histo_opts)

histoxs_scaled = {}
histoxs = {}
kappas = [0.001, 0.01, 0.1, 1, 10]

for kappa in kappas:
    sde_copy = sdes.SDE(2, init, drift, diffusion, boundaries, split)
    sde_copy.set_parameters(parameters | {'kappa': kappa})
    histoxs[kappa] = histogramx.copy(f"_kappa={kappa}", last_kwargs={'sde': sde_copy})
    histo_subs0 = histoxs[kappa][0]
    histo_subs1 = histoxs[kappa][1]
    lnode = LambdaNode("l_kappa={kappa}", histoxs[kappa], callback=lambda x, y, *_: (x / np.sqrt(kappa), y * np.sqrt(kappa)))
    print("hs data0", histo_subs0.data)
    print("hs data1", histo_subs1.data)
    print("l data", lnode.data)
    histoxs_scaled[kappa] = RealScatterNode("sc_kappa={kappa}", {'x': lnode[0], 'y': lnode[1]}, plot='scaled')

import proplot as pplt
format_kappas = NodeFigureFormat(
                subplots={'ncols': 2},
                fig_format={'yscale': 'log', 'xlabel': '$x$', 'ylabel':'Particle number density', 'yformatter': 'log', 'suptitle': 'Only diffusion, $u=0$, no shock. Comparing scaling (Ostrowski 1988)'}, 
                axs_format=[{'title': 'Not rescaled'}, {'title': 'Rescaled with $\sqrt{\kappa}$'}],
                legends_kw={1: {'loc': 'ur', 'ncols': 1}}
        )

nfig_kappa_double = NodeFigure(format_kappas)
for i, kappa in enumerate(kappas):
    kappa = float(kappa)
    nfig_kappa_double.add(histoxs[kappa], 0, plot_on="unscaled", label=f"$\kappa={kappa:.2}$")
    nfig_kappa_double.add(histoxs_scaled[kappa], 1, plot_on="scaled", label=f"$\kappa={kappa:.2}$")#, label="test")
nfig_kappa_double.savefig(figdir + '/' + name + '-kappa-double.pdf')
