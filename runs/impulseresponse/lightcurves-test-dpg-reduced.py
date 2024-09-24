from unicodedata import normalize
import sdesolver as sdes

import inspect
import time

from concurrent.futures import ProcessPoolExecutor, Future

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

def boundaries(t, x, L):
    ###
    # return 0
    ###

    x_a = carray(x, (2,))
    if np.abs(x_a[0]) > L:
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

format_lc = NodeFigureFormat(
                subplots={'ncols': 1, 'nrows': 1, 'figsize': (2.5, 2.5) },
                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Light curves'},
                axs_format=[{'xscale': 'linear', 'xlabel': '$T$', 'ylabel': 'Flux (a.u.)'}],
                legends_kw=[{'loc': 'lr', 'ncols': 1}]
            )

format_2meshes = NodeFigureFormat(
        subplots={'ncols': 2, 'sharex' : True},#, 'proj': '3d'},
                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Impulseresponse in 2D'},
                axs_format=[{'xscale': 'linear', 'xlabel': '$T$', 'ylabel': '$p/p_\\textrm{inj}$'}] * 2,
                legends_kw=[{'loc': 'ur', 'ncols': 1}] 
        )

format_hist_sed = NodeFigureFormat(
        subplots={'array': [[1, 1, 1, 2, 2, 2, 2, 2]], 'figsize': (8, 3), 'ncols': 1 },
        fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Time evolution of SEDs'},
                axs_format=[{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$E$ in eV', 'ylabel': 'Flux (a.u.)'}] * 2,
                legends_kw=[{'loc': 'ur', 'ncols': 1}],
        )


def get_greens(sde, sde_parameters, histo_opts, bins=None, cache=None, sourcefunction=None):
    observation_times = sde_parameters['observation_times']
    dt = sde_parameters['dt']
    x0 = sde_parameters['x0']
    n_particle = sde_parameters['n_particle']
    T = sde_parameters['T']

    bin_count = histo_opts['bin_count']

    if sourcefunction is None:
        logging.warning("Using delta injection since no source function was given")
        injection_times = [0.0] * n_particle
    else:
        injection_times, _ = samples_from_pdf(sourcefunction, n_particle, 0, x1=T)

    init_comp= [(inj_t, np.copy(x0)) for inj_t in injection_times]
    sde_injection = sde.copy(initial_condition=init_comp)
    solvernode = SDESolverNode('solver_sde', sde=sde_injection, scheme=b'euler', timestep=dt, observation_times=observation_times, nthreads=64, cache=cache)

    valuesp = {}
    min_v = np.inf
    max_v = -np.inf
    for T_ in observation_times:
        valuesp[T_] = SDEValuesNode(f'valuesp_sde_{T_}', {'x' : solvernode['solution'][T_]['x'], 'weights': solvernode['solution'][T_]['weights']}, index=1, T=T_, cache=cache)
        vdata = valuesp[T_]['values'].data
        if not len(vdata) == 0:
            min_v = min(min_v, min(vdata))
            max_v = max(max_v, max(vdata))

    if bins is None:
        bins = np.logspace(np.log10(min_v), np.log10(max_v), bin_count + 1)

    histop = {} 
    histop_density = {} 
    for T_, valuesp_ in valuesp.items():
        histop[T_] = HistogramNode(f'histop_sde_{T_}', {'values' : valuesp_['values'], 'weights' : valuesp_['weights']}, log_bins=True, normalize=None, **(histo_opts | {'cache': cache, 'edges': bins}))
        histop_density[T_] = HistogramNode(f'histop_sde_density{T_}', {'values' : valuesp_['values'], 'weights' : valuesp_['weights']}, log_bins=True, normalize='density', **(histo_opts | {'cache': cache, 'edges': bins}))

    greens = GreensFunction(f"green-sde", histop, plot='mesh', cache=cache, ignore_cache=False, plot_kwargs={'label': '$\\log N$', 'levels': 20, 'transpose': True})

    return greens, histop_density, histop, bins

def get_greens_sed(histop, sed_parameters, lower_cut=1e-6*u.erg/(u.second*u.centimeter**2), cache=None):
    # FIXME calculating all SEDs with the same n0 from density-normalized histograms is potentially wrong, since the histograms are all normalized with a different constant.
    p_inj = sed_parameters['p_inj']
    n0 = sed_parameters['n0']
    B = sed_parameters['B']
    source_size = sed_parameters['source_size']
    energy_range = sed_parameters['energy_range']

    worker_pool = ProcessPoolExecutor(mp_context=multiprocessing.get_context('fork'))

    seds = {}
    for T_, histop_ in histop.items():
        edists = HistogramElectronEnergyDistribution(f'edist_{T_}', parents={'histogram': histop_}, plot='naimadist', cache=cache, p_inj=p_inj, n0=n0, ignore_cache=True)
        synnaima = SynchrotronNaima(f'synnaima_{T_}', parents={'electron_distribution': edists}, plot="naimasrc", cache=cache, energy_range=energy_range, B=B, threadpool=worker_pool)
        sscnaima = SSCNaima(f'sscnaima_{T_}', parents={'electron_distribution': edists}, plot="naimasrc", cache=cache, energy_range=energy_range, B=B, source_size=source_size, threadpool=worker_pool)
        sumnaimas = SEDSum(f'sedsum_{T_}', parents=[synnaima, sscnaima], plot=["naimasum", "naimasrc"] , plot_kwargs={'linestyle': 'dotted', 'color': 'k'}, cache=cache, ignore_cache=False, set_ylim=False)
        sumnaimas_lim = LambdaNode(f'l_{T_}', parents=sumnaimas, cache=cache, 
                                       callback=lambda a: (a[0], u.Quantity([s if s > lower_cut else 0 * u.erg / (u.second * u.meter**2) for s in a[1]]), a[2]),
                                       ignore_cache=True)
        seds[T_] = NaimaAsHistogram(f"naimahist_{T_}", parents=sumnaimas_lim, cache=cache, plot="naima", show_errors=False, ignore_cache=False, plot_kwargs={'color': pplt.Colormap('haline')(1 - T_/40.0)})

    greens_sed = GreensFunction("green_sed", seds, plot='mesh', cache=cache, ignore_cache=False, plot_kwargs={'label': '$\\log N$', 'levels': 20, 'transpose': True})

    return greens_sed, seds

def impulse_response_comparison(source_name, sde, sde_parameters, histo_opts, bins, greens, sourcefunction=None):
    """
    Calculate the same solution by using the above greens function and SDE directly (changing its injection)
    """
    observation_times = sde_parameters['observation_times']
    dt = sde_parameters['dt']

    # continuous injection SDE for comparison
    cache_test = PickleNodeCache(g.cachedir, f"{g.name}_{source_name}_comparison")
    histo_opts_this = histo_opts | {'cache': cache_test, 'label': 'SDE initial condition', 'edges': bins}

    # colvolution
    if sourcefunction is None:
        greens_convolved = greens
        green_plot_group = 'mesh'
    else:
        greens_convolved = InjectionConvolveHistogram(f"testinj_{source_name}", greens, plot='inj', cache=cache_test, source_callback=sourcefunction, ignore_cache=False)
        green_plot_group = 'inj'

    greens_sdeinjection, histop_sdeinjection, _, _ = get_greens(sde, sde_parameters, histo_opts_this, bins=bins, cache=cache_test, sourcefunction=sourcefunction)

    histop_convolved = {}
    histop_sde = {}
    for T_ in observation_times:
        histop_convolved[T_] = ValueExtract(f"pl_convolved{T_}", greens_convolved, plot='pl', cache=cache_test, T=T_, normalize='density', plot_kwargs={'label': f"con T={T_}", 'linewidth': 0.7}, ignore_cache=False)
        histop_sde[T_] = ValueExtract(f"pl_sde{T_}", greens_sdeinjection, plot='pl', cache=cache_test, T=T_, normalize='density', plot_kwargs={'label': f"con sde T={T_}", 'linewidth': 0.7}, ignore_cache=False)


    # Greens functions (meshes)
    nfig3d = NodeFigure(format_2meshes)
    nfig3d.add(greens_convolved, 0, plot_on=green_plot_group)
    nfig3d.add(greens_sdeinjection, 1, plot_on='mesh')
    nfig3d.savefig(f"{g.figdir}/{g.name}_{source_name}_comparison_greens.pdf")
    #nfig3d.show_nodes("nodes.pdf")

    # Histograms
    ncols = int(np.sqrt(len(histop_convolved)))
    format4 = NodeFigureFormat(
                   subplots={'ncols': ncols, 'nrows': int(len(histop_convolved) / ncols + 0.5) },
                   fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Impulseresponse single plots'},
                   axs_format=[{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$', 'ylabel': '$N$'}] * len(histop_convolved),
                   legends_kw=[{'loc': 'ur', 'ncols': 1}] * (len(histop_convolved) )
            )

    nfig = NodeFigure(format4)
    for i, T_ in enumerate(observation_times):
        nfig.add(histop_convolved[T_], i, plot_on='pl')
        nfig.add(histop_sde[T_], i, plot_on='pl')
        nfig.add(histop_sdeinjection[T_], i, plot_on='histos')
    nfig.savefig(f"{g.figdir}/{g.name}_{source_name}_comparison_histop.pdf")

def impulse_response_lightcurves(source_name, greens, greens_sed, lightcurve_ranges, spectra_times=[], sourcefunction=None):
    # continuous injection SDE for comparison
    cache_test = PickleNodeCache(g.cachedir, f"{g.name}_{source_name}_lightcurves")

    # colvolution
    if sourcefunction is None:
        greens_sed_convolved = greens_sed
        greens_convolved = greens
        green_plot_group = 'mesh'
    else:
        greens_sed_convolved = InjectionConvolveHistogram(f"greens_sed_convolved_{source_name}", greens_sed, plot='inj', cache=cache_test, source_callback=sourcefunction, ignore_cache=False)
        greens_convolved = InjectionConvolveHistogram(f"greens_convolved_{source_name}", greens, plot='inj', cache=cache_test, source_callback=sourcefunction, ignore_cache=False)
        green_plot_group = 'inj'

    powerlaws = {}
    seds = {}
    cycle = iter(pplt.Cycle('default'))
    for T_ in spectra_times:
        this_color = next(cycle)['color']
        powerlaws[T_] = ValueExtract(f"powerlaw_{T_}", greens_convolved, plot='pl', cache=cache_test, T=T_, normalize='density', plot_kwargs={'label': f"T={T_}", 'linewidth': 0.7, 'color': this_color}, ignore_cache=False)
        seds[T_] = ValueExtract(f"sed_{T_}", greens_sed_convolved, plot='pl', cache=cache_test, T=T_, normalize=None, plot_kwargs={'label': f"T={T_}", 'linewidth': 0.7, 'color': this_color}, ignore_cache=False)

    lightcurves = {}
    for E in lightcurve_ranges:
        if isinstance(E, Real):
            Estr = f"{E:.2g}"
        else:
            Estr = f"{E[0]:.2g}-{E[1]:.2g}" 

        lightcurves[E] = TimeSeriesExtract(f"lightcurve_{Estr}", greens_sed_convolved, plot='lc', cache=cache_test, v=E, normalize=None, plot_kwargs={'label': f'$E={Estr}\\mathrm{{eV}}$'})


    # Greens functions (meshes)
    nfig3d = NodeFigure(format_2meshes)
    nfig3d.add(greens_convolved, 0, plot_on=green_plot_group)
    nfig3d.add(greens_sed_convolved, 1, plot_on=green_plot_group)
    nfig3d.savefig(f"{g.figdir}/{g.name}_{source_name}_lightcurves_greens.pdf")

    # Lightcurves
    nfig_lc = NodeFigure(format_lc)
    for i, (_, lc) in enumerate(lightcurves.items()):
        nfig_lc.add(lc, 0, plot_on='lc')
    nfig_lc.savefig(f"{g.figdir}/{g.name}_{source_name}_lightcurves.pdf")

    # Powerlaws and SEDs
    nfig_hist_sed = NodeFigure(format_hist_sed)
    for T_ in spectra_times:
        nfig_hist_sed.add(powerlaws[T_], 0, plot_on='pl')
        nfig_hist_sed.add(seds[T_], 1, plot_on='pl')
    nfig_hist_sed.savefig(f'{g.figdir}/{g.name}_{source_name}_lightcurves_histos_seds.pdf')


    # ncols = int(np.sqrt(len(seds)))
    # format4 = NodeFigureFormat(
    #                subplots={'ncols': ncols, 'nrows': int(len(seds) / ncols + 0.5) },
    #                fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Impulseresponse single plots'},
    #                axs_format=[{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$', 'ylabel': '$N$'}] * len(seds),
    #                legends_kw=[{'loc': 'ur', 'ncols': 1}] * (len(seds) )
    #         )
    #
    # nfig = NodeFigure(format4)
    # for i, (_, sed) in enumerate(seds.items()):
    #     nfig.add(sed, i, plot_on='naimasrc')
    # nfig.savefig(f"{figdir}/{name}_seds.pdf")



class Globals:
    cachedir = "cache"
    figdir = "figures"

    name = "impulseresponse-lc-dpg-reduced"

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

    histo_opts = {
            'plot' : 'histos',
            'ignore_cache' : False,
            'label': 'T={T}',
            'bin_count' : 60,
            }

    sed_parameters = {
            'p_inj' : 1e2 * constants.m_e * constants.c, #1e1
            'n0' : 1e52,
            'B' : 5e1 * u.G, # 1e1 mG
            'source_size' : 0.2 * u.pc, # 1e3 * u.pc
            'energy_range' : np.logspace(-5, 11.5, 100) * u.eV,
            }

    T = 2000.0 
    sde_parameters = {
            'dt' : 0.001,
            'T' : T,
            'n_particle' : 1000,
            'x0' : np.array([0.0, 1.0]),
            'observation_times' : np.linspace(0.0, T, 41),
            }

    xdiff = (parameters['a'] + parameters['b']) * np.sqrt(parameters['q'] * sde_parameters['dt'])
    parameters['L'] = 2500 * xdiff

    print("cutoff", cutoff_webb84(*webb84_from_params(parameters)))
    parameters |= {'cutoff' : cutoff_webb84(*webb84_from_params(parameters))}

    sde = sdes.SDE([(0.0, [0.0, 1.0])], drift, diffusion, boundaries, split)
    sde.set_parameters(parameters)

    cache = PickleNodeCache(cachedir, f"{name}_green")


    lc_ranges = [(200*1e-6, 400*1e-6), (4e0, 6e0), (0.5e6, 1.5e6), (1e9, 3e9)]
    #lc_ranges = [300*1e-6, 4e1, 1e6, 2e9]
    spectra_times = [60, 100, 200, 500, 2000]

    greens, histop_density, histop, bins = get_greens(sde, sde_parameters, histo_opts, cache=cache)
    greens_sed, seds = get_greens_sed(histop, sed_parameters, cache=cache)

    # Histograms
    ncols = int(np.sqrt(len(histop)))
    format4 = NodeFigureFormat(
                   subplots={'ncols': ncols, 'nrows': int(len(histop) / ncols + 0.5) },
                   fig_format={'yscale': 'log', 'yformatter': 'log', 'figtitle': 'Impulseresponse single plots'},
                   axs_format=[{'xscale': 'log', 'xformatter': pplt.SciFormatter(), 'xlabel': '$p/p_\\textrm{inj}$', 'ylabel': '$N$'}] * len(histop),
                   legends_kw=[{'loc': 'ur', 'ncols': 1}] * (len(histop) )
            )

    nfig = NodeFigure(format4)
    for i, T_ in enumerate(sde_parameters['observation_times']):
        nfig.add(histop[T_], i, plot_on='histos')
    nfig.savefig(f"{figdir}/{name}_histop.pdf")


g = Globals()
nfig3d = NodeFigure(format_2meshes)
nfig3d.add(g.greens, 0, plot_on='mesh')
#nfig3d.add(greens_sdeinjection, 1, plot_on='mesh')
nfig3d.savefig(f"{g.figdir}/{g.name}_greens.pdf")

#impulse_response_comparison("delta", g.sde, g.sde_parameters, g.histo_opts, g.bins, g.greens)
impulse_response_lightcurves("delta", g.greens, g.greens_sed, g.lc_ranges, g.spectra_times)

#impulse_response_comparison("const", g.sde, g.sde_parameters, g.histo_opts, g.bins, g.greens, sourcefunction=np.vectorize(lambda x: 1)) 
impulse_response_lightcurves("const", g.greens, g.greens_sed, g.lc_ranges, g.spectra_times, sourcefunction=np.vectorize(lambda x: 1))

#impulse_response_comparison("const_stop", g.sde, g.sde_parameters, g.histo_opts, g.bins, g.greens, sourcefunction=np.vectorize(lambda x: 1 if x < g.T / 2 else 0)) 
impulse_response_lightcurves("const_stop", g.greens, g.greens_sed, g.lc_ranges, g.spectra_times, sourcefunction=np.vectorize(lambda x: 1 if x < g.T / 2 else 0)) 

