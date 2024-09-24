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
                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Impulseresponse in 2D'},
                axs_format=[{'xscale': 'linear', 'xlabel': '$T$', 'ylabel': '$p/p_\\textrm{inj}$'}],
                legends_kw=[{'loc': 'ur', 'ncols': 1}] 
        )


cachedir = "cache"
figdir = "figures"

name = "impulseresponse-lc-test"


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

histo_opts = {'plot' : 'histos', 'cache' : cache, 'ignore_cache' : False, 'label': 'T={T}'}


valuesp = {}
histogramp = {}
seds = {}
edists = {}
edists1 = {}
edists2 = {}
sumnaimas = {}
sumnaimas_lim = {}
bins = None
energy_range = np.logspace(-5, 14, 100) * u.eV
for T_ in obs_at[::-1]:
    #if T_ < 40.0:
    #    continue
    valuesp[T_] = SDEValuesNode(f'valuesp_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=1, T=T_, cache=cache,
        confine_range=[(0, -confine_x, confine_x)],
    )
    if bins is None:
        vdata = valuesp[T_]['values'].data
        bins = np.logspace(np.log10(min(vdata)), np.log10(max(vdata)), bin_count + 1)

    histogramp[T_] = HistogramNode(f'histop_sameedges_{T_}', {'values' : valuesp[T_]['values'], 'weights' : valuesp[T_]['weights']}, log_bins=True, normalize=None, **(histo_opts | {'edges' : bins}))
    edists[T_] = HistogramElectronEnergyDistribution(f'edist_{T_}', parents={'histogram': histogramp[T_]}, plot='naimadist', cache=cache, p_inj = 1e-1 * constants.m_e * constants.c, n0=1e52) #p=1e1
    #edists1[T_] = HistogramElectronEnergyDistribution(f'edist1_{T_}', parents={'histogram': histogramp[T_]}, plot='naimadist', cache=cache, p_inj = 1e2 * constants.m_e * constants.c, n0=1e52)
    #edists2[T_] = HistogramElectronEnergyDistribution(f'edist2_{T_}', parents={'histogram': histogramp[T_]}, plot='naimadist', cache=cache, p_inj = 1e1 * constants.m_e * constants.c, n0=1e52)
    synnaima = SynchrotronNaima(f'synnaima_{T_}', parents={'electron_distribution': edists[T_]}, plot="naimasrc", cache=cache, energy_range=energy_range, B=1e3*u.mG) #B=1e1 mG
    sscnaima = SSCNaima(f'sscnaima_{T_}', parents={'electron_distribution': edists[T_]}, plot="naimasrc", cache=cache, energy_range=energy_range, B=1e3*u.G, source_size=10 * u.pc) #source = 1e3 pc
    sumnaimas[T_] = SEDSum(f'sedsum_{T_}', parents=[synnaima, sscnaima], plot="naimasrc", plot_kwargs={'linestyle': 'dotted', 'color': 'k'}, cache=cache)
    sumnaimas_lim[T_] = LambdaNode(f'l_{T_}', parents=sumnaimas[T_], cache=cache, 
                                   callback=lambda a: (a[0], u.Quantity([s if s > 1e-15 * u.erg / (u.second * u.meter**2) else 0 * u.erg / (u.second * u.meter**2) for s in a[1]]), a[2]),
                                   ignore_cache=True)
    #total_flux = LambdaNode(f'tflux_{T_}', parents=sumnaima[2], callback=lambda x: np.sum(x))
    #seds[T_] = NodeGroup(f'naimagroup_{T_}', parents=[sumnaima[0], sumnaima[2], None, PassiveNode(f"p_{T_}", obj=energy_range, cache=cache)]
    seds[T_] = NaimaAsHistogram(f"naimahist_{T_}", parents=sumnaimas_lim[T_], cache=cache, plot="naima", show_errors=False, ignore_cache=False, plot_kwargs={'color': pplt.Colormap('haline')(T_/40.0)})
    #print(T_, seds[T_].data[1])

#print(sumnaimas[T].data)
#print(seds[T].data)


greens = GreensFunction("green", histogramp, plot='mesh', cache=cache, ignore_cache=False, plot_kwargs={'label': '$\\log N$', 'levels': 20, 'transpose': True})
greens_sed = GreensFunction("green_sed", seds, plot='mesh', cache=cache, ignore_cache=False, plot_kwargs={'label': '$\\log N$', 'levels': 20, 'transpose': True})

def impulse_response_test(source_name, sourcefunction):
    # continuous injection SDE for comparison
    cache_test = PickleNodeCache(cachedir, f"{name}_{source_name}")

    injection_times, inj_norm = samples_from_pdf(sourcefunction, n_particle, 0, x1=T)
    histogram_inj = HistogramNode(f'histo_inj', {'values' : PassiveNode('injt', obj=injection_times)}, log_bins=False, normalize='width', plot=True, cache=cache_test)

    sourcefunction_norm = lambda x: sourcefunction(x)# * inj_norm

    init_comp= [(inj_t, np.copy(x0)) for inj_t in injection_times]

    sde_comp = sdes.SDE(init_comp, drift, diffusion, boundaries, split)
    sde_comp.set_parameters(parameters)

    solvernode_comp = SDESolverNode('solver_comp', sde=sde_comp, scheme=b'euler', timestep=dt, observation_times=[T], nthreads=64, cache=cache_test, splitted=True)

    valuesx_comp = SDEValuesNode(f'valuesx_comp', {'x' : solvernode_comp['solution'][T]['x'], 'weights': solvernode_comp['solution'][T]['weights']}, index=0, T=T, cache=cache_test)
    valuesp_comp = SDEValuesNode(f'valuesp_comp', {'x' : solvernode_comp['solution'][T]['x'], 'weights': solvernode_comp['solution'][T]['weights']}, index=1, T=T, cache=cache_test,
        confine_range=[(0, -confine_x, confine_x)],
    )
    histogramx_comp = HistogramNode(f'histox_comp', {'values' : valuesx_comp['values'], 'weights' : valuesx_comp['weights']}, log_bins=False, normalize='width', **(histo_opts | {'cache': cache_test}))
    histogramp_comp = HistogramNode(f'histop_comp', {'values' : valuesp_comp['values'], 'weights' : valuesp_comp['weights']}, log_bins=True, normalize='density', **(histo_opts | {'cache': cache_test, 'label': 'SDE initial condition', 'edges': bins}))

    # colvolution
    test_inj = InjectionConvolveHistogram("testinj", greens, plot='inj', cache=cache_test, source_callback=sourcefunction_norm, ignore_cache=False)
    test_pl = ValueExtract("testpl", test_inj, plot='pl', cache=cache_test, T=T, normalize='density', plot_kwargs={'label': "Convolved impulse response", 'linewidth': 0.7})
    test_lc = TimeSeriesExtract("testlc", test_inj, plot='pl', cache=cache_test, v=10, normalize='none')

    # plotting 
    nfig_inj = NodeFigure(format_lc)
    nfig_inj.add(histogram_inj, 0, plot_on='histos')
    nfig_inj.savefig(f"{figdir}/{name}_{source_name}_injection.pdf")

    nfig_inj = NodeFigure(format_comparison)
    nfig_inj.add(test_pl, 0, plot_on='pl')
    nfig_inj.add(histogramp_comp, 0, plot_on='histos')
    nfig_inj.add(Residual("res", parents=[test_pl, histogramp_comp[:2]], plot=True), 1)
    nfig_inj.savefig(f"{figdir}/{name}_{source_name}_Tconst.pdf")

    nfig_inj = NodeFigure(format_lc)
    nfig_inj.add(test_lc, 0, plot_on='pl')
    nfig_inj.savefig(f"{figdir}/{name}_{source_name}_pconst.pdf")

# nfig3d = NodeFigure(format_mesh)
# nfig3d.add(greens, 0, plot_on='mesh')
# nfig3d.savefig(f"{figdir}/{name}_greens.pdf")

nfig3d_2 = NodeFigure(format_mesh)
nfig3d_2.add(greens_sed, 0, plot_on='mesh')
nfig3d_2.savefig(f"{figdir}/{name}_greens_sed.pdf")

ncols = int(np.sqrt(len(seds)))
format4 = NodeFigureFormat(
       subplots={'ncols': ncols, 'nrows': int(len(seds) / ncols + 0.5) },
               fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Impulseresponse single plots'},
               axs_format=[{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$', 'ylabel': '$N$'}] * len(seds),
               legends_kw=[{'loc': 'ur', 'ncols': 1}] * (len(seds) )
        )

nfig = NodeFigure(format4)
for i, (_, h) in enumerate(seds.items()):
    nfig.add(h, i, plot_on='naima')
nfig.savefig(figdir + '/' + name + '_seds.pdf')

nfig = NodeFigure(format_pl)
for i, (_, h) in enumerate(seds.items()):
    nfig.add(h, 0, plot_on='naima')
nfig.savefig(figdir + '/' + name + '_seds_single.pdf')


nfig = NodeFigure(format4)
for i, (_, h) in enumerate(histogramp.items()):
    nfig.add(h, i, plot_on='histos')
nfig.savefig(figdir + '/' + name + '_histos.pdf')

# nfig = NodeFigure(format4)
# for i, (_, h) in enumerate(seds.items()):
#     nfig.add(h, i, plot_on='naimadist')
# nfig.savefig(figdir + '/' + name + '_dists.pdf')


# nfig = NodeFigure(format4)
# for i, ((_, h), (_, j), (_, k)) in enumerate(zip(edists.items(), edists1.items(), edists2.items())):
#     nfig.add(h, i, plot_on='naimadist')
#     nfig.add(j, i, plot_on='naimadist')
#     nfig.add(k, i, plot_on='naimadist')
# nfig.savefig(figdir + '/' + name + '_pdist.pdf')

#nfig_ir = NodeFigure(format_pl)
#for i, (_, h) in enumerate(histogramp.items()):
#    if i % 5 == 0:
#        nfig_ir.add(h, 0, plot_on='histos')
#nfig_ir.savefig(f"{figdir}/{name}_ir.pdf")
#
#impulse_response_test("const", np.vectorize(lambda x: 1)) 
#impulse_response_test("power", lambda x: x**2 + 1)
#impulse_response_test("inverse", lambda x: 1 / (x + 1))
