import numpy as np
import proplot as pplt
from matplotlib.lines import Line2D
import logging
import time
import argparse
import inspect

import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.special.kruells import *
from pybatch.special.sourcetest import *

from evaluation.experiment import *
from evaluation.helpers import *
from evaluation.extractors import *
from evaluation.exporters import *

import astropy.units as u
import astropy.constants as constants
from astropy.coordinates import Distance

import formats
import chains
from node.nodefigure import NodeFigure
from node.cache import PickleNodeCache
from node.special import *
from node.node import *


logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('node.node').setLevel(logging.DEBUG)

# for new chain based experiments
figdir = 'figures/ex'
cachedir = 'pickle/ex'

# for old extractor based experiments
cache_opts = {
    'cachedir' : 'pickle',
    'regenerate' : False
}
store_opts = {
    'dir' : 'out',
    'format' : 'pdf'
}

""" *********** """
""" Source Test """

# This test needs to use an integrator but there is no Node for this atm
#def ex_sourcetest():
#    param = { 'Tmax' : 1,
#              'x_min' : -1,
#              'x_max' : 1,
#              'x0' : 0,
#              'N' : 500000
#            }
#
#    exset = Experiment(PyBatchSourcetest, param)
#    return exset.run()
#
#test_sourcetest = ExporterHist(
#        ex_sourcetest,
#        store_opts,
#        log_x=False,
#        log_y=False,
#        use_integrator=0,
#        bin_count=200
#    )


""" *********** """
""" Kruells 1 """

def kruells1():
    name = inspect.currentframe().f_code.co_name

    param = { 'Tesc' : 0.25,
              'x0' : 0,
              'p0' : 0,
              'N' : 50
            }

    times = np.array([64, 200, 640, 1000]) / 100

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells1, cache, param, times, np.inf)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

""" *********** """
""" Kruells 2 """

def kruells2():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              'r_inj' : 0.1,
              'k_syn' : 0,#.0001,
              'dx_inj' : 1,
              'x0' : 0,
              'u0' : 0,
              'N' : 50000
            }

    times = np.array([64, 200, 640, 1000])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells2, cache, param, times, bin_count=50, confine_x=np.inf)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')
    
""" *********** """
""" Kruells 2a (synchrotron) """

def kruells2a():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              'r_inj' : 0.1,
              'k_syn' : 0.0001,
              'dx_inj' : 1,
              'x0' : 0,
              'u0' : 0,
              'N' : 50000
            }

    times = np.array([64, 200, 640, 1000])


    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells2, cache, param, times, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')
    
""" *********** """
""" Kruells 3 """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection

    b: wider timerange, powerlaws
"""

def kruells3():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
            }

    times = np.array([20, 64, 200, 640])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells3, cache, param, times, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')
    
""" *********** """
""" Kruells 3a """
"""
    Synchrotron

"""

def kruells3a():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.003,
              'k_syn' : 0.0005,
              'x0' : 0,
              'y0' : 1,
            }

    times = np.array([20, 64, 200, 640])


    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells3, cache, param, times, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')


""" *********** """
""" Kruells 4 """
""" Integrator -> chain not possible atm """

@cached(**cache_opts)
def ex_kruells4():

    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              'a_inj' : 100,
              'k_syn' : 0,#.0001,
              'r_inj' : 100,
              'N' : 50000,
              'x0' : 0,
              'y0' : 1
            }

    times = np.array([64, 200, 640, 1000])
    
    exset = ExperimentSet(PyBatchKruells4, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells4 = ExporterDoubleHist(
        ex_kruells4,
        store_opts,
        log_x=(False, True),
        log_y=(True, True),
        use_integrator=0,
        bin_count=50
    )

""" *********** """
""" Kruells 5 """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection
    (like kruells 923)
    but with a spatially variable diffusion coefficient

    parametrisation of 1994 paper

"""

def kruells5():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.003,
              'beta_s' : 1,
              'k_syn' : 0,
              'x0' : 0,
              'y0' : 1,
              'q' : 1
            }

    times = np.array([20, 64, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells5, cache, param, times, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

""" *********** """
""" Kruells 6a """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection

    parametrisation of 1992 paper

    synchrotron losses
"""

def kruells6a():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.05,
              'k_syn' : 0.0005,
              'x0' : 0,
              'u0' : 0
            }

    times = np.array([64, 200, 640, 1000])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells6, cache, param, times, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')


""" *********** """
""" Kruells 6b """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection

    parametrisation of 1992 paper

    b: wider timerange, powerlaws
"""

def kruells6b():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'u0' : 1,
            }

    times = np.array([20, 64, 200, 640, 1000, 2000])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells6, cache, param, times, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

""" *********** """
""" Kruells 6c """
"""
    u0 = 0
"""

def kruells6c():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.03,
              'k_syn' : 0,#.001,
              'x0' : 0,
              'u0' : 0,
            }

    times = np.array([20, 64, 200, 640])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells6, cache, param, times, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

""" *********** """
""" Kruells 7 """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection
    (like kruells 923)
    but with a spatially variable diffusion coefficient

    parametrisation of 1992 paper

"""

def kruells7():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'u0' : 0,
              'q' : 1
            }

    times = np.array([20, 64, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells7, cache, param, times, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

def kruells7a():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'u0' : 0,
              'q' : 10
            }

    times = np.array([20, 64, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells7, cache, param, times, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

def kruells7b():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'u0' : 0,
              'q' : 0.01
            }

    times = np.array([20, 64, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells7, cache, param, times, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')
    
# this one is for the talk at FRANCI 21
def kruells7c():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.0003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'u0' : 0,
              'q' : 1
            }

    times = np.array([6.4, 20, 64, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells7, cache, param, times, confine_x=5, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
7c with a single normalisation on all spatial histograms
Single normalisation is not available any more and makes little sense
"""

def kruells7c1():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.0003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'u0' : 0,
              'q' : 1
            }

    times = np.array([6.4, 20, 64, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells7, cache, param, times, confine_x=5, bin_count=30)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
7c1 with more detail
"""

def kruells7c2():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.00003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'u0' : 0,
              'q' : 1
            }

    times = np.array([6.4, 20, 64, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells7, cache, param, times, confine_x=5, bin_count=30)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')


# this one is for the talk at FRANCI 21 too
def kruells7d():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 0.9,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.0003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'u0' : 0,
              'q' : 1
            }

    times = np.array([2, 6.4, 20, 64, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells7, cache, param, times, confine_x=1, bin_count=50)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')
    
# this one is for the talk at FRANCI 21 too
def kruells7e():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'kappa' : 5,
              'beta_s' : 0.9,
              'r' : 4,
              'dt' : 0.1,
              't_inj' : 0.0001,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'u0' : 0,
              'q' : 1
            }

    times = np.array([6.4, 20, 64, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells7, cache, param, times, bin_count=50, confine_x=1)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')


""" *********** """
""" Kruells 8 """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection
    (like kruells 923)
    but with a spatially variable diffusion coefficient

    parametrisation of 1992 paper

"""

# also using an integrator -> not chainable atm
@cached(**cache_opts)
def ex_kruells8():
    param = { 'Xsh' : 0.25,
              'beta_s' : 1,
              'r' : 4,
              'dt' : 0.1,
              'r_inj' : 100,
              'a_inj' : 100,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
              'q' : 1,
              'N' : 50000
            }

    times = np.array([20, 64, 200])
    
    exset = ExperimentSet(PyBatchKruells8, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells8 = ExporterDoubleHistPL(
        ex_kruells8,
        store_opts,
        log_y=(True, True),
        #average_bin_size=200,
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "p/p_inj"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, True),
        powerlaw_annotate=True,
        xlim=((None, None), (None, None)),
        use_integrator=0
)

"""
like 7, but with 94 parametrisation
"""

def kruells9():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.002,
              'beta_s' : 0.06,
              'r' : 4,
              'dt' : 0.001,
              't_inj' : 0.0003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
              'q' : 1
            }

    times = np.array([0.64, 2.0, 6.4, 20])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells9, cache, param, times, confine_x=0.05, bin_count=30)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')
    
"""
more detail
"""

def kruells9a():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.002,
              'beta_s' : 0.06,
              'r' : 4,
              'dt' : 0.001,
              't_inj' : 0.0003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
              'q' : 1
            }

    times = np.array([0.64, 2.0, 6.4, 20, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells9, cache, param, times, confine_x=0.05, bin_count=40)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
t_inj = dt
"""
def kruells9a1():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.002,
              'beta_s' : 0.06,
              'r' : 4,
              'dt' : 0.001,
              't_inj' : 0.2,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
              'q' : 1
            }

    times = np.array([0.64, 2.0, 6.4, 20, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells9, cache, param, times, confine_x=0.05, bin_count=30)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
generate synchrotron spectrum
"""
def kruells9b():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 1.5,
              'beta_s' : 0.9,
              'r' : 4,
              'dt' : 0.8,
              't_inj' : 0.01,
              'k_syn' : 0.005,
              'x0' : 0,
              'y0' : 1,
              'q' : 5
            }
    # xdiff = 1.8, xadv = 1.17, a (Webb, Kruells) = approx 9

    times = np.array([200, 2000, 5000])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells9, cache, param, times, confine_x=100, bin_count=30, plot_on="spectra")
    transform = MomentumCount('mc', [powerlaw.parents['dataset']], plot=False, cache=cache, p_inj=100*constants.m_e*constants.c)
    
    nu_range = np.logspace(10, 20, 100) * u.Hz
    model_params = dict(delta_D=10, z=Distance(1e27, unit=u.cm).z, B=1 * u.G, d_L=1e27 * u.cm, R_b=1e16 * u.cm)
    radiation_params = dict(plot=True, model_params=model_params, nu_range=nu_range, gamma_integrate=np.logspace(1, 9, 20), cache=cache)
    synchrotronflux = SynchrotronExactAgnPy('synchro', {'N_data' : transform}, **radiation_params)
    synchrotronfluxdelta = SynchrotronDeltaApproxAgnPy('synchrodelta', {'N_data' : transform}, **radiation_params)
    sscflux = SynchrotronSelfComptonAgnPy('ssc', {'N_data' : transform}, **radiation_params)

    nfig = NodeFigure(formats.doublehistSED)
    nfig.add(histosetx, 0, plot_on="spectra")
    nfig.add(histosetp, 1, plot_on="spectra")
    nfig.add(powerlaw, 1, plot_on="spectra")
    nfig.add(synchrotronflux, 2)
    nfig.add(synchrotronfluxdelta, 2)
    nfig.add(sscflux, 2)
    nfig[2].format(ylim=(1e-14, 1e-11))
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
generate synchrotron spectrum
different k_syn
"""
def kruells9b1():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 1.5,
              'beta_s' : 0.9,
              'r' : 4,
              'dt' : 0.8,
              't_inj' : 0.01,
              'k_syn' : 0.005,
              'x0' : 0,
              'y0' : 1,
              'q' : 5,
              'Tmax' : 2000
            }
    # xdiff = 1.8, xadv = 1.17

    ksyn = [0, 0.0001, 0.0005]
    param_sets = {'k_syn={}'.format(ks) : {'param' : param | {'k_syn': ks}} for ks in ksyn}

    cache = PickleNodeCache(cachedir, name)
    print(isinstance(cache, KwargsCacheMixin))
    batch = BatchNode('batch', batch_cls=PyBatchKruells9, cache=cache, ignore_cache=False)
    points = PointNode('points', {'batch' : batch}, cache=cache, ignore_cache=False)

    points_range = {}
    for n, kw in param_sets.items():
        points_range[n] = {'points' : points.copy(n, last_kwargs=kw)}

    valuesx = ValuesNode('valuesx', index=0, cache=cache, ignore_cache=False)
    valuesp = ValuesNode('valuesp', index=1, cache=cache, ignore_cache=False,
            confine_range=[(0, -100, 300)],
            confinements=[(0, lambda x: np.abs(x) <= 100)]
        )

    histo_opts = {'bin_count' : 50, 'plot' : 'spectra', 'cache' : cache, 'ignore_cache' : False, 'label' : '$k_\\mathrm{{syn}}={k_syn}$'} 
    histogramx = HistogramNode('histox', {'values' : valuesx}, log_bins=False, normalize='width', **histo_opts)
    histogramp = HistogramNode('histop', {'values' : valuesp}, log_bins=True, normalize='density', **histo_opts)
    
    nu_range = np.logspace(10, 25, 100) * u.Hz
    p_inj = 100 * constants.m_e * constants.c
    gamma_integrate = np.logspace(1, 9, 20)
    model_params = dict(delta_D=10, z=Distance(1e27, unit=u.cm).z, B=1 * u.G, d_L=1e27 * u.cm, R_b=1e16 * u.cm)

    def cb(model_params, batch_params):
        k_syn = batch_params['k_syn']
        B = (constants.m_e * constants.c / constants.e.si)**2 * np.sqrt(6 * np.pi * constants.eps0 * k_syn)
        B_cgs = B * np.sqrt(4 * np.pi / constants.mu0)
        return model_params# | {'B' : B_cgs}

    radiation_params = dict(plot=True, model_params=model_params, model_params_callback=cb, nu_range=nu_range, gamma_integrate=gamma_integrate, cache=cache, ignore_cache=False)
    transform = MomentumCount('mc', [histogramp], plot=False, cache=cache, p_inj=p_inj)
    synchrotronflux = SynchrotronExactAgnPy('synchro', {'N_data' : transform}, **radiation_params)
    synchrotronfluxdelta = SynchrotronDeltaApproxAgnPy('synchrodelta', {'N_data' : transform}, plot_kwargs={'linestyle': 'dashed', 'alpha': 0.6}, **radiation_params)
    sscflux = SynchrotronSelfComptonAgnPy('ssc', {'N_data' : transform}, plot_kwargs={'linestyle': 'dotted'}, **radiation_params)
    fluxes = NodeGroup('fluxgroup', [synchrotronflux, synchrotronfluxdelta, sscflux])

    histosetx = copy_to_group('groupx', histogramx, last_parents=points_range)
    fluxset = copy_to_group('groupflux', fluxes, last_parents=points_range)
    histosetp = NodeGroup('groupp', fluxset.search_parents_all('histop'))

    nfig = NodeFigure(formats.doublehistSED)
    nfig.add(histosetx, 0, plot_on="spectra")
    nfig.add(histosetp, 1, plot_on="spectra")
    nfig.add(fluxset, 2)
    #nfig.show_nodes("synchronodes.pdf")
    #nfig.add(synchrotronfluxdelta, 2)
    #nfig.add(sscflux, 2)
    nfig[2].format(ylim=(1e-14, 1e-8))
    nfig[2].legend(ncols=1, loc='r', handles=[
        Line2D([], [], label='Synchrotron', color='k'),
        Line2D([], [], linestyle='dashed', alpha=0.6, label='SSC (delta approx.)', color='k'),
        Line2D([], [], linestyle='dashdot', label='SSC', color='k')])
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
dkappa/dx from eq.(19)
"""
def kruells12():
    name = inspect.currentframe().f_code.co_name
    param = { 
              't_inj' : 2,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
            }

    # judging from the dt study, 3000 timesteps should be enough
    param_calc, param_num = chains.param_from_numerical(dx_adv=0.1, delta=0.5, sigma=0.5, beta_s=0.01, r=4, n_timesteps=3000)
    param |= param_calc
    param['Xdiff'] = 4 * param_num['dx_diff']
    
    cache = PickleNodeCache(cachedir, name)
    #pcache = PointsNodeCache(cachedir, name)
    histogramx, histogramp = chains.get_chain_single(PyBatchKruells12, cache, param=param, confine_x=1, bin_count=30)
    #histogramx.search_parent('points').cache = pcache
    #histogramx.search_parent('points').ignore_cache = True
    powerlaw = PowerlawNode('pl', {'dataset' : histogramp }, plot=True)

    for v in histogramx.search_parent_all('valuesx') + histogramp.search_parent_all('valuesp'):
        v.ignore_cache = True

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histogramx, 0)
    nfig.add(histogramp, 1)
    nfig.add(powerlaw, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
dkappa/dx from eq.(19)
"""
def kruells12ts():
    name = inspect.currentframe().f_code.co_name
    param = { 
              't_inj' : 0.2,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
            }

    # judging from the dt study, 3000 timesteps should be enough
    param_calc, param_num = chains.param_from_numerical(dx_adv=0.1, delta=0.5, sigma=0.5, beta_s=0.01, r=4, n_timesteps=3000)
    param |= param_calc
    param['Xdiff'] = 4 * param_num['dx_diff']
    
    times = [1000, 5000, 12000, 30000, 50000] #Tmax=15000 <-> 3000 timesteps
    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells12, cache, param, times, confine_x=1, bin_count=30)

    for _, histop in histosetp.parents_iter:
        old = histop.ext['plot_kwargs']
        histop.set(plot_kwargs=old | {'alpha' :0.6})

    nfig = NodeFigure(formats.doublehist, suptitle="Particle distributions using $d\\kappa /d x = -\\kappa / X_\\textrm{diff}$")
    nfig[1].format(xformatter=pplt.AutoFormatter())
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(powerlaw, 1, plot_kwargs={'color': 'red'})
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
dkappa/dx from eq.(19) (other sign)
"""
def kruells13ts():
    name = inspect.currentframe().f_code.co_name
    param = { 
              't_inj' : 0.2,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
            }

    # judging from the dt study, 3000 timesteps should be enough
    param_calc, param_num = chains.param_from_numerical(dx_adv=0.1, delta=0.5, sigma=0.5, beta_s=0.01, r=4, n_timesteps=3000)
    param |= param_calc
    param['Xdiff'] = 4 * param_num['dx_diff']
    
    times = [1000, 5000, 12000, 30000, 50000] #Tmax=15000 <-> 3000 timesteps
    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells13, cache, param, times, confine_x=1, bin_count=30)

    for _, histop in histosetp.parents_iter:
        old = histop.ext['plot_kwargs']
        histop.set(plot_kwargs=old | {'alpha' :0.6})

    nfig = NodeFigure(formats.doublehist, suptitle="Particle distributions using $d\\kappa /d x = \\kappa / X_\\textrm{diff}$")
    nfig[1].format(xformatter=pplt.AutoFormatter(), xlocator=[1, 2, 3])
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(powerlaw, 1, plot_kwargs={'color': 'red'})
    nfig.show_nodes("test.pdf")
    nfig.savefig(figdir + '/' + name + '.pdf')

""" *********** """
""" Kruells B1 """
""" Reproduce 1994/Fig. 4
    using continous pseudo particle injection
    (like kruells 923)

    parametrisation of 1992 paper

"""
def kruellsB1():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'gamma' : 0.2,
              'r' : 3,
              'dt' : 0.01,
              't_inj' : 0.1,
              'k_syn' : 0.005,
              'x0' : 0,
              'y0' : 1,
              'a1' : 9,
            }

    times = np.array([640, 2000])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruellsB1, cache, param, times, confine_x=10, bin_count=30)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

def kruellsB1a():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'gamma' : 0.2,
              'r' : 3,
              'dt' : 0.01,
              't_inj' : 0.001,
              'k_syn' : 0.005,
              'x0' : 0,
              'y0' : 1,
              'a1' : 9,
            }

    times = np.array([64, 200, 640])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruellsB1, cache, param, times, confine_x=5, bin_count=30)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

def kruellsB1b():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'gamma' : 0.2,
              'r' : 3,
              'dt' : 0.01,
              't_inj' : 0.002,
              'k_syn' : 0.005,
              'x0' : 0,
              'y0' : 1,
              'a1' : 9,
            }

    times = np.array([64, 200, 640])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruellsB1, cache, param, times, confine_x=5, bin_count=30)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

def kruellsB1c():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.25,
              'gamma' : 0.2,
              'r' : 3,
              'dt' : 0.01,
              't_inj' : 0.0004,
              'k_syn' : 0.005,
              'x0' : 0,
              'y0' : 1,
              'a1' : 9,
            }

    times = np.array([64, 200, 640])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruellsB1, cache, param, times, confine_x=5, bin_count=100)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    #xlim=((None, None), (10**-1, 10**1)),
    #ylim=((None, None), (10**-4, 10**1)),
    nfig.savefig(figdir + '/' + name + '.pdf')
    
def kruellsC1a():
    name = inspect.currentframe().f_code.co_name
    param = { 'kappa' : 1,
              'a2' : 0.01,
              'k_syn' : 0,
              'dt' : 0.004,
              't_inj' : 0.0004,
              'x0' : 0,
              'y0' : 1,
              'Lx' : 20,
              'Lylower' : 0,
              'Lyupper' : 200,
            }

    times = np.array([64, 640])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruellsC1, cache, param, times, confine_x=0.2, bin_count=200)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    #xlim=((None, None), (10**-1, 10**2)),
    #ylim=((None, None), (10**-3, 10**0)),
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
Achterberg 2011 model where D = q * V (in Kruells terms kappa = q * beta)
"""
def achterberg1():
    name = inspect.currentframe().f_code.co_name
    param_sets = {'A': {'param' : { 
                          'V' : 0.02,
                          'dt' : 0.01,
                          'r' : 4,
                          'q' : 12,
                          'Ls' : 0.032,
                          't_inj' : 2,
                          'x0' : 0,
                          'y0' : 1,
                          'Tmax' : 1000
                        },
                        'label_fmt_fields' : {
                          'name' : 'A'
                        }},
                  'B': {'param' : { 
                          'V' : 0.02,
                          'dt' : 0.01,
                          'r' : 4,
                          'q' : 5,
                          'Ls' : 0.02,
                          't_inj' : 2,
                          'x0' : 0,
                          'y0' : 1,
                          'Tmax' : 1000
                        },
                        'label_fmt_fields' : {
                          'name' : 'B'
                        }},
                  'C': {'param' : { 
                          'V' : 0.02,
                          'dt' : 0.01,
                          'r' : 4,
                          'q' : 20,
                          'Ls' : 0.04,
                          't_inj' : 2,
                          'x0' : 0,
                          'y0' : 1,
                          'Tmax' : 1000
                        },
                        'label_fmt_fields' : {
                          'name' : 'C'
                        }},
                  'C2': {'param' : { 
                          'V' : 0.02,
                          'dt' : 0.01,
                          'r' : 4,
                          'q' : 20,
                          'Ls' : 0.04,
                          't_inj' : 2,
                          'x0' : 0,
                          'y0' : 1,
                          'Tmax' : 10000
                        },
                        'label_fmt_fields' : {
                          'name' : 'C2'
                        }},
                  'D': {'param' : { 
                          'V' : 0.02,
                          'dt' : 0.01,
                          'r' : 4,
                          'q' : 0.2,
                          'Ls' : 0.004,
                          't_inj' : 2,
                          'x0' : 0,
                          'y0' : 1,
                          'Tmax' : 1000
                        },
                        'label_fmt_fields' : {
                          'name' : 'D'
                        }},
                  'D2': {'param' : { 
                          'V' : 0.02,
                          'dt' : 0.01,
                          'r' : 4,
                          'q' : 0.2,
                          'Ls' : 0.004,
                          't_inj' : 20,
                          'x0' : 0,
                          'y0' : 1,
                          'Tmax' : 10000
                        },
                        'label_fmt_fields' : {
                          'name' : 'D'
                        }}
            }

    cache = PickleNodeCache(cachedir, name)
    histosetx, powerlaws = chains.get_chain_parameter_series(
        PyBatchAchterberg1,
        cache,
        param_sets=param_sets,
        confine_x=10,
        bin_count=30,
        powerlaws=True,
        histo_opts={'label' : '{name}'}
    )

    #for v in histogramx.search_parent_all('valuesx') + histogramp.search_parent_all('valuesp'):
    #    v.ignore_cache = True

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(powerlaws, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
Achterberg 2011 model where D = q * V (in Kruells terms kappa = q * beta)
Timeseries D param set from achterberg1
"""
def achterberg2():
    name = inspect.currentframe().f_code.co_name
    param = {'V' : 0.02,
                  'dt' : 0.01,
                  'r' : 4,
                  'q' : 0.2,
                  'Ls' : 0.004,
                  't_inj' : 20,
                  'x0' : 0,
                  'y0' : 1
                 }

    times = [500, 5000, 10000]
    param_sets = chains.generate_timerange_set(param, times, constant_particle_count=500)
    cache = PickleNodeCache(cachedir, name)
    histosetx, powerlaws = chains.get_chain_parameter_series(
        PyBatchAchterberg1,
        cache,
        param_sets=param_sets,
        confine_x=50,
        bin_count=30,
        powerlaws=True,
        histo_opts={'label' : '$T_\\textrm{{max}}={Tmax}$'}
    )

    for v in histosetx.search_parents_all('valuesx') + powerlaws.search_parents_all('valuesp'):
        v.ignore_cache = True

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(powerlaws, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

    nfig = NodeFigure(formats.momentumhist, title='$L_\\textrm{diff} = \\frac{D}{V} = 0.2 = \\textrm{const.}$\\\\(Achterberg/Schure Fig. 2, one datapoint $\\epsilon = 0.02$)')
    for v in powerlaws.search_parents_all('histop'):
        v.set(plot_kwargs={'alpha': 0.5})
    nfig.add(powerlaws, 0)
    nfig.savefig(figdir + '/' + name + '-momentum.pdf')



""" ***************** """
""" Run experiment(s) """
if __name__ == '__main__':
    #kruells6b()
    #kruells3()
    #kruells3a()
    #kruells2()
    #kruells2a()
    #kruells2a2()
    #kruells5()
    #kruells8()
    #kruells4()
    #kruells7c2()
    #kruells7c1()
    #kruells7c()
    #kruells7d()
    #kruells7e()
    #kruells7a()
    #kruells7b()
    #kruells6a()
    #kruells6c()
    #kruells9b()
    kruells9b1()
    #kruells9a()
    #kruells9a1()

    #kruellsB1()
    #kruellsB1a()
    #kruellsB1b()
    #kruellsB1c()

    #kruellsC1a()
    #kruells12ts()
    #kruells12()
    #kruells13ts()

    #achterberg2()
