import sdesolver as sdes

import inspect
import time

from grapheval.nodefigure import NodeFigure, NodeFigureFormat
from grapheval.cache import PickleNodeCache
from grapheval.node import *
from grapheval.graph import draw_node_chain

from src.basicnodes import *
from src.newnodes import *
from src.impulseresponse import *
from src.sampling import samples_from_pdf

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
    ###
    return 0
    ###

    x_a = carray(x, (2,))
    if np.abs(x_a[0]) > 0.005:
        return 1
    else:
        return 0

def nosplit(t, x, last_t, last_x):
    return False

def split(t, x, last_t, last_x, w):
    if x[1] / last_x[1] >= 1.8:#1.41:
        return True
    else:
        return False

import proplot as pplt

format_comparison = NodeFigureFormat(
        subplots={'ncols': 1, 'nrows': 2 , 'sharex': True},
                fig_format={'figtitle': 'Two ways to calculate arbitrary injection functions'},
                axs_format=[{'yformatter': 'log', 'yscale': 'log', 'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '', 'ylabel': '$N$'},
                            {'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$', 'ylabel': '$\Delta N/N$'}],
                legends_kw=[{'loc': 'ur', 'ncols': 1}, {'loc': 'ur', 'ncols': 1}],
        )

format_pl = NodeFigureFormat(
        subplots={'ncols': 1 },
                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Two ways to calculate arbitrary injection functions'},
                axs_format=[{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$', 'ylabel': '$N$'}],
                legends_kw=[{'loc': 'ur', 'ncols': 1}],
        )

format_lc = NodeFigureFormat(
        subplots={'ncols': 1 },
                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Comparison of runs with and without split particles'},
                axs_format=[{'xscale': 'linear', 'xformatter': pplt.SciFormatter(), 'xlabel': '$T$', 'ylabel': '$N$'}],
                legends_kw=[{'loc': 'ur', 'ncols': 1}],
        )

format_mesh = NodeFigureFormat(
        subplots={'ncols': 1},#, 'proj': '3d'},
                fig_format={'yscale': 'log', 'figtitle': 'Impulseresponse in 2D'},
                axs_format=[{'xscale': 'linear', 'xlabel': '$T$', 'ylabel': '$p/p_\\textrm{inj}$'}],
                legends_kw=[{'loc': 'ur', 'ncols': 1}] 
        )


cachedir = "cache"
figdir = "figures"

name = "impulseresponse-direct"


#T = 200.0
#t_inj = 0.05
T = 40.0
t_inj = 0.001
x0 = np.array([0.0, 1.0])
dt = 0.001
confine_x=100
n_particle = int(T / t_inj) 

parameters = {
        'Xsh' : 0.001215, 
        'a' : 0.0375,
        'b' : 0.0225,
        'k_syn' : 0,
        'q' : 1,
    }

# calculate impulse response
init = [(0.0, np.copy(x0)) for i in range(n_particle)]

sde = sdes.SDE(init, drift, diffusion, boundaries, split)
sde.set_parameters(parameters)

cache = PickleNodeCache(cachedir, f"{name}_green")

obs_at = np.linspace(0.0, T, 41)
solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=64, cache=cache, splitted=True)

bin_count = 60

histo_opts = {'plot' : 'histos', 'cache' : cache, 'bin_count': bin_count, 'ignore_cache' : False, 'label': 'T={T}'}

valuesp_green = {}
for T_ in obs_at[::-1]:
    valuesp_green[T_] = SDEValuesNode(f'valuesp_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=1, T=T_, cache=cache,
        confine_range=[(0, -confine_x, confine_x)],
    )

green = GreensFunctionValues('green', parents=valuesp_green, cache=cache)

def impulse_response_test(source_name, sourcefunction):
    # continuous injection SDE for comparison
    cache_test = PickleNodeCache(cachedir, f"{name}_{source_name}")

    injection_times, inj_norm = samples_from_pdf(sourcefunction, n_particle, 0, x1=T)

    init_comp= [(inj_t, np.copy(x0)) for inj_t in injection_times]

    sde_comp = sdes.SDE(init_comp, drift, diffusion, boundaries, split)
    sde_comp.set_parameters(parameters)

    solvernode_comp = SDESolverNode('solver_comp', sde=sde_comp, scheme=b'euler', timestep=dt, observation_times=[T], nthreads=64, cache=cache_test, splitted=True)

    valuesx_comp = SDEValuesNode(f'valuesx_comp', {'x' : solvernode_comp['solution'][T]['x'], 'weights': solvernode_comp['solution'][T]['weights']}, index=0, T=T, cache=cache_test)
    valuesp_comp = SDEValuesNode(f'valuesp_comp', {'x' : solvernode_comp['solution'][T]['x'], 'weights': solvernode_comp['solution'][T]['weights']}, index=1, T=T, cache=cache_test,
        confine_range=[(0, -confine_x, confine_x)],
    )
    histogramx_comp = HistogramNode(f'histox_comp', {'values' : valuesx_comp['values'], 'weights' : valuesx_comp['weights']}, log_bins=False, normalize='width', **(histo_opts | {'cache': cache_test}))
    histogramp_comp = HistogramNode(f'histop_comp', {'values' : valuesp_comp['values'], 'weights' : valuesp_comp['weights']}, log_bins=True, normalize='density', **(histo_opts | {'cache': cache_test, 'label': 'SDE initial condition'}))

    # colvolution
    convolution = InjectionConvolveValuesDirect("convol", parents=valuesp_green, cache=cache_test, source_callback=sourcefunction)
    histogramp_convolution = HistogramNode(f'histop_convol', {'values' : convolution[T]['values'], 'weights' : convolution[T]['weights']}, log_bins=True, normalize='density', **(histo_opts | {'cache': cache_test, 'label': 'Convolution direct'}))

    # indicrect convolution
    convolution_indir = InjectionConvolveValues("convol_indir", parents=green, cache=cache_test, source_callback=sourcefunction)
    histogramp_convolution_indir = HistogramNode(f'histop_convol_indir', {'values' : convolution_indir[T]['values'], 'weights' : convolution_indir[T]['weights']}, log_bins=True, normalize='density', **(histo_opts | {'cache': cache_test, 'label': 'Convolution indirect'}))

    # plotting 
    nfig_inj = NodeFigure(format_comparison)
    nfig_inj.add(histogramp_convolution, 0, plot_on='histos')
    nfig_inj.add(histogramp_convolution_indir, 0, plot_on='histos')
    nfig_inj.add(histogramp_comp, 0, plot_on='histos')
    #nfig_inj.add(Residual("res", parents=[test_pl, histogramp_comp[:2]], plot=True), 1)
    nfig_inj.savefig(f"{figdir}/{name}_{source_name}-comparison.pdf")

#ncols = int(np.sqrt(len(histogramp)))
#format4 = NodeFigureFormat(
#        subplots={'ncols': ncols, 'nrows': int(len(histogramp) / ncols + 0.5) },
#                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Impulseresponse single plots'},
#                axs_format=[{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$', 'ylabel': '$N$'}] * len(histogramp),
#                legends_kw=[{'loc': 'ur', 'ncols': 1}] * (len(histogramp) )
#        )
#
#nfig = NodeFigure(format4)
#for i, (_, h) in enumerate(histogramp.items()):
#    nfig.add(h, i, plot_on='histos')
#nfig.savefig(figdir + '/' + name + '.pdf')

#nfig_ir = NodeFigure(format_pl)
#for i, (_, h) in enumerate(histogramp_sameedges.items()):
#    if i % 5 == 0:
#        nfig_ir.add(h, 0, plot_on='histos')
#nfig_ir.savefig(f"{figdir}/{name}_ir.pdf")

impulse_response_test("const", np.vectorize(lambda x: 1)) 
impulse_response_test("power", lambda x: x**2 + 1)
impulse_response_test("inverse", lambda x: 1 / (x + 1))
