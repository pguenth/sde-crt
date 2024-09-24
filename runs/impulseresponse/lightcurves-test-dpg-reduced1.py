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


from src.radiativenodes import *
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
from astropy import units as u
from astropy import constants
import proplot as pplt

#warnings.simplefilter("error", np.VisibleDeprecationWarning)
logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)


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

def diffusion(out, t, x, Xsh, a, b, q):
    # cpp: kruells_shockaccel2_diffusion
    diffval = np.sqrt(2.0 * q * (a + b)**2)

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

def split(t, x, last_t, last_x, w, split_lim1, split_lim2, cutoff):
    if x[1] > cutoff and x[1] / last_x[1] >= split_lim2:
        return True
    if x[1] / last_x[1] >= split_lim1:#1.41:
        return True
    else:
        return False


def webb84_from_params(params):
    r = (params['a'] + params['b']) / (params['a'] - params['b'])
    a1 = 1 / (4 * params['q'] * params['k_syn'])
    a2 = a1 / r**2

    return r, a1, a2

def cutoff_webb84(r, a1, a2):
    # fig. 3 of webb1984
    return 4 / 3 * (r - 1) / (r / a1 + 1 / a2)


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

format_seds = NodeFigureFormat(
        subplots={'figsize': (5, 3), 'ncols': 1 },
        fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Time evolution of SEDs'},
                axs_format=[{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$E$ in eV', 'ylabel': 'Flux (a.u.)'}],
                legends_kw=[{'loc': 'ur', 'ncols': 1}],
        )

format_mesh = NodeFigureFormat(
        subplots={'ncols': 1},#, 'proj': '3d'},
                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Impulseresponse in 2D'},
                axs_format=[{'xscale': 'linear', 'xlabel': '$T$', 'ylabel': '$p/p_\\textrm{inj}$'}],
                legends_kw=[{'loc': 'ur', 'ncols': 1}] 
        )


cachedir = "cache"
figdir = "figures"

name = "impulseresponse-lc-dpg-new2"


#T = 200.0
#t_inj = 0.05
T = 2000.0
n_particle = 1000
t_inj = T / n_particle
x0 = np.array([0.0, 1.0])
dt = 0.001
confine_x=100

parameters = {
        'Xsh' : 0.25, 
        'a' : 0.596,
        'b' : 0.298,
        'k_syn' : 1e-5,
        'q' : 5.556,
        'split_lim1' : 1.65,
        'split_lim2' : 1.065,
    }

print("cutoff", cutoff_webb84(*webb84_from_params(parameters)))
parameters |= {'cutoff' : cutoff_webb84(*webb84_from_params(parameters))}

# calculate impulse response
init = [(0.0, np.copy(x0)) for i in range(n_particle)]

sde = sdes.SDE(init, drift, diffusion, boundaries, split)
sde.set_parameters(parameters)

cache = PickleNodeCache(cachedir, f"{name}_green")

obs_at = np.linspace(0.0, T, 41)
solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=64, cache=cache, splitted=True)

bin_count = 60

histo_opts = {'plot' : 'histos', 'cache' : cache, 'ignore_cache' : True, 'label': 'T={T}'}


valuesp = {}
histogramp = {}
min_v = np.inf
max_v = -np.inf

for T_ in obs_at:
    valuesp[T_] = SDEValuesNode(f'valuesp_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=1, T=T_, cache=cache, ignore_cache=True,
        # confine_range=[#(0, -confine_x, confine_x),
        #                (1, 0, 10**2)
        #                ],
    )
    min_v = min(min_v, min(valuesp[T_]['values'].data))
    max_v = max(max_v, max(valuesp[T_]['values'].data))

bins = np.logspace(np.log10(min_v), np.log10(max_v), bin_count + 1)

for T_ in obs_at:
    histogramp[T_] = HistogramNode(f'histop_sameedges_{T_}', {'values' : valuesp[T_]['values'], 'weights' : valuesp[T_]['weights']}, log_bins=True, normalize='density', **(histo_opts | {'edges' : bins}))
    print("valuesp", T_, len(valuesp[T_]['values'].data))

greens = GreensFunction("green", histogramp, plot='mesh', cache=cache, ignore_cache=False, plot_kwargs={'label': '$\\log N$', 'levels': 20, 'transpose': True})

def impulse_response_test(source_name, sourcefunction=None):
    # continuous injection SDE for comparison
    cache_test = PickleNodeCache(cachedir, f"{name}_{source_name}")

    if not sourcefunction is None:
        injection_times, _ = samples_from_pdf(sourcefunction, n_particle, 0, x1=T)
    else:
        logging.warning("Using delta injection since no source function was given")
        injection_times = [0.0] * n_particle

    #histogram_inj = HistogramNode(f'histo_inj', {'values' : PassiveNode('injt', obj=injection_times)}, log_bins=False, normalize='width', plot=True, cache=cache_test)

    sourcefunction_norm = lambda x: sourcefunction(x)# * inj_norm

    init_comp= [(inj_t, np.copy(x0)) for inj_t in injection_times]
    
    sde_comp = sdes.SDE(init_comp, drift, diffusion, boundaries, split)
    sde_comp.set_parameters(parameters)

    solvernode_comp = SDESolverNode('solver_comp', sde=sde_comp, scheme=b'euler', timestep=dt, observation_times=obs_at, nthreads=64, cache=cache_test, splitted=True)

    #valuesx_comp = SDEValuesNode(f'valuesx_comp', {'x' : solvernode_comp['solution'][T]['x'], 'weights': solvernode_comp['solution'][T]['weights']}, index=0, T=T, cache=cache_test)
        #confine_range=[(0, -confine_x, confine_x)],
    #histogramx_comp = HistogramNode(f'histox_comp', {'values' : valuesx_comp['values'], 'weights' : valuesx_comp['weights']}, log_bins=False, normalize='width', **(histo_opts | {'cache': cache_test}))

    # colvolution
    if sourcefunction is None:
        test_inj = greens
    else:
        test_inj = InjectionConvolveHistogram("testinj", greens, plot='inj', cache=cache_test, source_callback=sourcefunction_norm, ignore_cache=False)
    comparison_pls = {}
    for T_ in obs_at[1:]:
        valuesp_comp = SDEValuesNode(f'valuesp_comp_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=1, T=T_, cache=cache_test, ignore_cache=True)
        print("valuesp_comp", T_, len(valuesp_comp['values'].data))
        comparison_pls[T_] = HistogramNode(f'histop_comp_{T_}', {'values' : valuesp[T_]['values'], 'weights' : valuesp[T_]['weights']}, log_bins=True, normalize='density', **(histo_opts | {'cache': cache_test, 'label': 'SDE initial condition'}))#, 'edges': bins}))

    nfig3d = NodeFigure(format_mesh)
    nfig3d.add(test_inj, 0, plot_on='mesh')
    nfig3d.savefig(f"{figdir}/{name}_{source_name}_2d_convolved.pdf")

    ncols = int(np.sqrt(len(comparison_pls)))
    format4 = NodeFigureFormat(
           subplots={'ncols': ncols, 'nrows': int(len(comparison_pls) / ncols + 0.5) },
                   fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Impulseresponse single plots'},
                   axs_format=[{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$', 'ylabel': '$N$'}] * len(comparison_pls),
                   legends_kw=[{'loc': 'ur', 'ncols': 1}] * (len(comparison_pls) )
            )

    nfig = NodeFigure(format4)
    for i, ((T_, h), (T2_, hcomp)) in enumerate(zip(sorted(histogramp.items())[1:], sorted(comparison_pls.items()))):
        print(T_, T2_)
        assert T_ == T2_
        #print("extracted", T_, h.data)
        nfig.add(h, i, plot_on='histos')
        #nfig.add(h, i, plot_on='pl')
        nfig.add(hcomp, i, plot_on='histos')
    nfig.savefig(f"{figdir}/{name}_{source_name}_histop.pdf")
    


#impulse_response_test("const", sourcefunction=np.vectorize(lambda x: 1)) 
impulse_response_test("delta")
