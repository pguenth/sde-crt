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
    x_a = carray(x, (2,))
    if np.abs(x_a[0]) > 0.005:
        return 1
    else:
        return 0

def split(t, x, last_t, last_x):
    if x[1] / last_x[1] >= 1.8:#1.41:
        return True
    else:
        return False


cachedir = "cache"
figdir = "figures"

name = "9a1-nochains"


#T = 200.0
#t_inj = 0.05
T = 40.0
t_inj = 0.005
x0 = np.array([0.0, 1.0])
dt = 0.001
confine_x=100
n_particle = int(T / t_inj) 

#init = [(i * t_inj, np.copy(x0)) for i in range(n_particle)]
init = [(0.0, np.copy(x0)) for i in range(n_particle)]

parameters = {
        'Xsh' : 0.001215, 
        'a' : 0.0375,
        'b' : 0.0225,
        'k_syn' : 0,
        'q' : 1,
    }

sde = sdes.SDE(2, init, drift, diffusion, boundaries, split)
sde_nosplit = sdes.SDE(2, init, drift, diffusion, boundaries, nosplit)

sde.set_parameters(parameters)
sde_nosplit.set_parameters(parameters)

cache = PickleNodeCache(cachedir, name)

obs_at = [T/8, T/4, T / 2, T]
solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=4, cache=cache, splitted=True)
solvernode_nosplit = SDESolverNode('solver_nosplit', sde=sde_nosplit, scheme=b'euler', timestep=dt, observation_times=obs_at, cache=cache, splitted=False)

histo_opts = {'bin_count' : 30, 'plot' : True, 'cache' : cache, 'ignore_cache' : False, 'label': 'T={T}, splitted: {splitted}'}

valuesx = {}
valuesp = {}
histogramx = {}
histogramp = {}
for T_ in obs_at:
    valuesx[T_] = SDEValuesNode(f'valuesx_{T_}', {'x' : solvernode[T_]['x'], 'weights': solvernode[T_]['weights']}, index=0, T=T_, cache=cache)
    valuesp[T_] = SDEValuesNode(f'valuesp_{T_}', {'x' : solvernode[T_]['x'], 'weights': solvernode[T_]['weights']}, index=1, T=T_, cache=cache,
        confine_range=[(0, -confine_x, confine_x)],
    )
    histogramx[T_] = HistogramNode(f'histox_{T_}', {'values' : valuesx[T_]['values'], 'weights' : valuesx[T_]['weights']}, log_bins=False, normalize='width', **histo_opts)
    histogramp[T_] = HistogramNode(f'histop_{T_}', {'values' : valuesp[T_]['values'], 'weights' : valuesp[T_]['weights']}, log_bins=True, normalize='density', **histo_opts)


valuesx_nosplits = {}
valuesp_nosplits = {}
histogramx_nosplits = {}
histogramp_nosplits = {}
for T_ in obs_at:
    valuesx_nosplits[T_] = SDEValuesNode(f'valuesx_nosplit_{T_}', {'x' : solvernode_nosplit[T_]['x'], 'weights': solvernode_nosplit[T_]['weights']}, index=0, T=T_, cache=cache)
    valuesp_nosplits[T_] = SDEValuesNode(f'valuesp_nosplit_{T_}', {'x' : solvernode_nosplit[T_]['x'], 'weights': solvernode_nosplit[T_]['weights']}, index=1, T=T_, cache=cache,
        confine_range=[(0, -confine_x, confine_x)],
    )
    histogramx_nosplits[T_] = HistogramNode(f'histox_nosplit_{T_}', {'values' : valuesx_nosplits[T_]['values'], 'weights' : valuesx_nosplits[T_]['weights']}, log_bins=False, normalize='width', **histo_opts)
    histogramp_nosplits[T_] = HistogramNode(f'histop_nosplit_{T_}', {'values' : valuesp_nosplits[T_]['values'], 'weights' : valuesp_nosplits[T_]['weights']}, log_bins=True, normalize='density', **histo_opts)


    #histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(NumpyBatch, cache, param, times, confine_x=0.05, bin_count=30)
    #histosetx.map_tree(lambda b : b.set(nthreads=8), "batch")

import proplot as pplt
format4 = NodeFigureFormat(
                subplots={'array': [[1, 2], [3, 4]]},
                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Comparison of runs with and without split particles'},
                axs_format=[{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$'}] * 4,
                legends_kw=[{'loc': 'ur', 'ncols': 1}] * 4
        )

nfig = NodeFigure(format4)
#nfig.add(histogramx, 0)
#nfig.add(histogramp, 1)
for i, (_, h) in enumerate(histogramp.items()):
    nfig.add(h, i)
for i, (_, h) in enumerate(histogramp_nosplits.items()):
    nfig.add(h, i)
#for _, h in histogramx.items(): 
#    nfig.add(h, 0)
#for _, h in histogramx_nosplits.items(): 
#    nfig.add(h, 0)
#nfig.add(powerlaw, 1)
nfig.savefig(figdir + '/' + name + '.pdf')

draw_node_chain(histogramp_nosplits[T], "splitgraph.pdf")


#import cProfile
#pr = cProfile.Profile()
#cProfile.run('kruells9a1_newstyle()', filename="test-cpp-propagation.perf")
