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

from agnpy import spectra

import formats
import chains
from toptygin import *
from grapheval.nodefigure import NodeFigure, NodeFigureFormat
from grapheval.cache import PickleNodeCache
from src.specialnodes import *
from grapheval.node import *
from grapheval.graph import draw_node_chain

import multiproc

pplt.rc['text.latex.preamble'] = r"""
        \usepackage{siunitx}
	\sisetup{
		per-mode = reciprocal,
		separate-uncertainty = true,
		locale = US,
		output-decimal-marker = {.},
                round-mode = figures,
                round-precision = 2
	}
        \DeclareSIUnit\gauss{G}
    """

# gauss/tesla equivalency
gauss_tesla_eq = (u.G, u.T, lambda x: x / np.sqrt(4 * np.pi / constants.mu0), lambda x: x * np.sqrt(4 * np.pi / constants.mu0))

logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#logging.getLogger('node.node').setLevel(logging.DEBUG)

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
dxadv(') = 0.000534
dxdiff = 0.00189
-> sigma > 1?
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
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells9, cache, param, times, confine_x=0.05, bin_count=40, histo_opts={'label' : '$T_\\textrm{{max}}={Tmax}$'})

    nfmt = NodeFigureFormat(base=formats.doublehist,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(powerlaw, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
sigma = 0.5
"""
def kruells9a1():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.001215,
              'beta_s' : 0.06,
              'r' : 4,
              'dt' : 0.001,
              't_inj' : 0.01,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
              'q' : 1
            }

    #times = np.array([0.64, 2.0, 6.4, 20, 200])
    times = np.array([0.2])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells9, cache, param, times, confine_x=0.05, bin_count=30)
    histosetx.map_tree(lambda b : b.set(nthreads=8), "batch")

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(powerlaw, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
using the numerical parameters
"""
def kruells9a2():
    name = inspect.currentframe().f_code.co_name
    param = { 
              't_inj' : 0.0035,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
            }

    # judging from the dt study, 3000 timesteps should be enough
    # invdelta = 3.1
    param_calc, _ = chains.param_from_numerical(dx_adv=0.1, delta=0.323, sigma=0.95, beta_s=1, r=4, n_timesteps=5000)
    param |= param_calc

    #times = int(param['Tmax'] * 10 / 20) * np.array([0.64, 2.0, 6.4, 20, 200]) / 10
    times = param['Tmax'] / 20 * np.array([0.64, 2.0, 6.4, 20, 200])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells9, cache, param, times, confine_x=50, bin_count=40, histo_opts={'label': '$T={Tmax:.2f}$'})
    histosetx.map_tree(lambda b : b.set(nthreads=8), "batch")
    histosetx.map_tree(lambda h : h.set(bin_width=1), "histo")
    histosetp.map_tree(lambda h : h.set(bin_width=1/30), "histo")
    #valuesx = histosetx.search_parent('valuesx_Tmax=' + str(max(times)))
    #valuesp = histosetx.search_parent('valuesp_Tmax=' + str(max(times)))
    #print(valuesx, valuesp)

    #histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=20, log_bins=(False, True), plot=True, style='contourf', normalize='none', log_histogram=True, cmap='Boreal')

    nfmt = NodeFigureFormat(base=formats.doublehistand2d)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    #nfig[0].format(xlim=(-30,100), ylim=(10, 3*1e3))
    nfig[1].format(ylim=(1e-4, 1e1))
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(powerlaw, 1)
    #nfig.add(histo2d, 2)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
using the numerical parameters
more detail
"""
def kruells9a3():
    name = inspect.currentframe().f_code.co_name
    param = { 
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
            }

    # judging from the dt study, 3000 timesteps should be enough
    # invdelta = 3.1
    param_calc, _ = chains.param_from_numerical(dx_adv=0.1, delta=0.323, sigma=0.95, beta_s=1, r=4, n_timesteps=5000)
    param |= param_calc
    print(param)

    times = np.array([3.6, 11.3, 36, 113, 1130])

    nparticles = 2e6

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells9, cache, param, times, confine_x=np.inf, bin_count=40, histo_opts={'label': '${Tmax:.1f}$'}, constant_particle_count=nparticles)
    def mod_b(b):
        b.set(nthreads=8)
        b.cache = None
        b.cache_not_found_regenerate = False
    
    def mod_h(h):
        h.set(bin_width=0.5)

    def mod_h2(h):
        T = float(h.name.split('=')[1])
        h.set(normalize='density', manual_normalization_factor=T)

    histosetx.map_tree(mod_b, "batch")
    histosetx.map_tree(lambda h: mod_h(h) or mod_h2(h), "histo")
    histosetp.map_tree(lambda h : h.set(bin_width=1/60), "histo")
    powerlaw.set(plot_kwargs=powerlaw.get_kwargs()['plot_kwargs'] | {'color': 'k'})
    powerlaw.set(ndigits=4)

    t2d = 113
    valuesx = histosetx.search_parent('valuesx_T=' + str(t2d))
    valuesp = histosetp.search_parent('valuesp_T=' + str(t2d))

    contourlevels = np.linspace(-8.5, -0.5, 21)
    histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=25, log_bins=(False, True), plot=True, 
                              style='contour', normalize='density', log_histogram=True, ignore_cache=False, cache=cache, label='${Tmax:.1f}$',
                              cmap='Haline', plot_kwargs=dict(labels=True, levels=contourlevels, cmap_kw={'reverse': True}, robust=True, labels_kw={'color' : 'gray'}))

    t2d_2 = 1130
    valuesx_2 = histosetx.search_parent('valuesx_T=' + str(t2d_2))
    valuesp_2 = histosetp.search_parent('valuesp_T=' + str(t2d_2))

    histo2d_2 = Histogram2DNode('histo2d_2', {'valuesx' : valuesx_2, 'valuesy' : valuesp_2}, bin_count=20, log_bins=(False, True), plot=True, cache=cache,
            style='contour', normalize='density', log_histogram=True, ignore_cache=False, limits=((-40, 50), (1, 200)), label='${Tmax:.0f}$',
                              linestyle='--', plot_kwargs=dict(labels=False, levels=contourlevels, color='k', alpha=0.45, robust=True, labels_kw={'color' : 'gray'}))

    #topt_param = {
    #               'x0': 0,
    #               'y0': 1, 
    #               'beta_s': 20, 
    #               'q': 0.21,
    #               'r': 4
    #             }

    #toptygin = ToptyginContourNode('topt', plot=True, cache=cache, params=topt_param, N0=0.4, T=7, x_range=(-40, 50), y_range=(0, 3), levels=contourlevels, detail=40)

    nfmt = NodeFigureFormat(base=formats.doublehistand2d2)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig.format(suptitle="Spatial and momentum spectrum of particles for pure diffusive shock acceleration")
    nfig[2].format(title="Contour plot of $\\log{{\\bar F}}$")
    nfig[0].format(xlim=(-30,100), ylim=(5e-4, 30))
    nfig[1].format(ylim=(5e-6, 1e1))
    nfig[2].format(xlim=(-40, 50), ylim=(1, 200))
    nfig[0].annotate('$\\delta =0.323,~~\\sigma=0.95$', (0.06, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(powerlaw, 1)
    nfig.add(histo2d, 2)
    nfig.add(histo2d_2, 2)
    handles_ts = []
    for n in histosetp.search_parents_all('histop'):
        handles_ts += n.handles
    nfig[1].legend(loc='ur', handles=handles_ts, ncols=1, title='Observation time $T$')
    nfig[1].legend(loc='ll', handles=powerlaw.handles,ncols=1, title='Powerlaw fit')
    #nfig.add(toptygin, 2)
    handles_contour = []
    for n in [histo2d, histo2d_2]:
        handles_contour += n.handles
    from matplotlib import lines
    handles_contour = [lines.Line2D([], [], color='k', label='${}$'.format(t2d))] + histo2d_2.handles 
    nfig[2].legend(loc='ul', handles=handles_contour, ncols=1, title='Observation time $T$')
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
= 9a but using the semiimplicitweakscheme2
"""
def kruells14():
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

    times = np.array([0.64, 2.0, 6.4, 20.0]) # 200

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells14, cache, param, times, confine_x=5, bin_count=40, histo_opts={'label' : '$T_\\textrm{{max}}={Tmax}$'}, log_bins=(False, True))

    nfmt = NodeFigureFormat(base=formats.doublehist,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(powerlaw, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
= 14 but some 2d stuff
with numerical comparison (wild)
this is somehow the best run by far despite delta=3.5, sigma>1 but dx_adv =0.00053
"""
def kruells14a():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.002,
              'beta_s' : 0.06,
              'r' : 4,
              'dt' : 0.001,
              't_inj' : 0.0003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
              'q' : 1,
              'Tmax' : 20.0
            } # this is 6 hours

    cache = PickleNodeCache(cachedir, name)
    batch = BatchNode('batch', batch_cls=PyBatchKruells14, cache=cache, param=param)
    points = PointNode('points', {'batch' : batch}, cache=cache)
    valuesx = ValuesNode('valuesx', {'points' : points}, index=0, cache=cache)
    valuesp = ValuesNode('valuesp', {'points' : points}, index=1, cache=cache)

    histox = HistogramNode('histox', {'values' : valuesx}, bin_count=40, log_bins=False, plot=True)
    histoy = HistogramNode('histoy', {'values' : valuesp}, bin_count=40, log_bins=True, plot=True)

    contourlevels = np.linspace(-4, 0, 12)
    histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=30, log_bins=(False, True), plot=True, 
                              style='contour', normalize='density', log_histogram=True, ignore_cache=True,
                              cmap='Haline', plot_kwargs=dict(labels=True, levels=contourlevels, cmap_kw={'reverse': True}, robust=True, labels_kw={'color' : 'gray'}))
    #histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=20, log_bins=(False, True), plot=True, style='contour', normalize='none', log_histogram=True, cmap='Boreal')

    topt_param = {
                   'x0': 0,
                   'y0': 1, 
                   'beta_s': 0.2, 
                   'q': 0.3,
                   'r': 4
                 }

    print(cache)
    print(isinstance(cache, KwargsCacheMixin))
    toptygin = ToptyginContourNode('topt', plot=True, cache=cache,
                    params=topt_param, N0=0.2, T=6,
                    x_range=(-0.4, 0.5), y_range=(0, np.log10(200)), 
                    levels=contourlevels, detail=60,
                    contour_opts=dict(color='k', linestyle='--', linewidth=0.5, labels=False))

    nfmt = NodeFigureFormat(base=formats.doublehistand2d)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig[2].format(xlim=(-0.4, 0.5), ylim=(1, 200))
    nfig.add(histox, 0)
    nfig.add(histoy, 1)
    nfig.add(histo2d, 2)
    nfig.add(toptygin, 2)
    #nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
= 14a retry
"""
def kruells14ax1():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.002,
              'beta_s' : 0.06,
              'r' : 4,
              'dt' : 0.001,
              't_inj' : 0.02,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
              'q' : 1,
              'Tmax' : 20.0
            }

    cache = PickleNodeCache(cachedir, name)
    batch = BatchNode('batch', batch_cls=PyBatchKruells14, cache=cache, param=param)
    points = PointNode('points', {'batch' : batch}, cache=cache)
    valuesx = ValuesNode('valuesx', {'points' : points}, index=0, cache=cache)
    valuesp = ValuesNode('valuesp', {'points' : points}, index=1, cache=cache)

    histox = HistogramNode('histox', {'values' : valuesx}, bin_count=40, log_bins=False, plot=True)
    histoy = HistogramNode('histoy', {'values' : valuesp}, bin_count=40, log_bins=True, plot=True)

    contourlevels = np.linspace(-4, 0, 12)
    histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=30, log_bins=(False, True), plot=True, 
                              style='contour', normalize='density', log_histogram=True, ignore_cache=True,
                              cmap='Haline', plot_kwargs=dict(labels=True, levels=contourlevels, cmap_kw={'reverse': True}, robust=True, labels_kw={'color' : 'gray'}))
    #histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=20, log_bins=(False, True), plot=True, style='contour', normalize='none', log_histogram=True, cmap='Boreal')

    topt_param = {
                   'x0': 0,
                   'y0': 1, 
                   'beta_s': 0.2, 
                   'q': 0.3,
                   'r': 4
                 }

    print(cache)
    print(isinstance(cache, KwargsCacheMixin))
    toptygin = ToptyginContourNode('topt', plot=True, cache=cache,
                    params=topt_param, N0=0.2, T=6,
                    x_range=(-0.4, 0.5), y_range=(0, np.log10(200)), 
                    levels=contourlevels, detail=60,
                    contour_opts=dict(color='k', linestyle='--', linewidth=0.5, labels=False))

    nfmt = NodeFigureFormat(base=formats.doublehistand2d)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig[2].format(xlim=(-0.4, 0.5), ylim=(1, 200))
    nfig.add(histox, 0)
    nfig.add(histoy, 1)
    nfig.add(histo2d, 2)
    nfig.add(toptygin, 2)
    #nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

def contourpl(T, lims, N0_corr, topt_param, N0_base, histoset, contourlevels, cache, topt_labels=False, labels_hist=True, cut_levels_topt=None, topt_detail=30, bins=30):
    N0 = N0_base / T * N0_corr
    points2d = histoset.search_parent('points_T=' + str(T))
    valuesx = ValuesNode('valuesx2d_T=' + str(T), parents={'points': points2d}, index=0, cache=cache, ignore_cache=False)
    valuesp = ValuesNode('valuesp2d_T=' + str(T), parents={'points': points2d}, index=1, cache=cache, ignore_cache=False)

    histo2d = Histogram2DNode('histo2d_T=' + str(T), {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=bins, log_bins=(False, True), plot=True, 
                              style='contour', normalize='density', log_histogram=True, ignore_cache=False, cache=cache, label='${Tmax:.1f}$',
                              cmap='Haline', plot_kwargs=dict(labels=labels_hist, levels=contourlevels, cmap_kw={'reverse': True}, robust=True,
                              labels_kw={'color' : 'gray'}), limits=lims)


    topt_contour = contourlevels if cut_levels_topt is None else contourlevels[cut_levels_topt:]
    toptygin = ToptyginContourNode('topt_T=' + str(T), plot=True, cache=cache,
                    params=topt_param, N0=N0, T=T,
                    x_range=lims[0], y_range=(np.log10(lims[1][0]), np.log10(lims[1][1])),
                    levels=topt_contour, detail=topt_detail,
                    contour_opts=dict(color='k', linestyle='--', label="Analytical solution", linewidth=0.5, alpha=1, labels=topt_labels))

    return histo2d, toptygin

def kruells14ax2_blueprint(name, num_params):
    param = { 
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
            }

    #param_calc, _ = chains.param_from_numerical(dx_adv=0.00053, delta=0.282, sigma=1.07, beta_s=0.06, r=4, n_timesteps=20000)
    param_calc, _ = chains.param_from_numerical(**num_params)
    param |= param_calc
    print(param)

    times = np.array([0.64, 2.0, 6.4, 20.0])

    #nparticles = 360000 # should be 8 hours...
    nparticles = 36000

    tmax = 200.0 #param['Tmax']

    do_splitted_max = False # if False do unsplitted rest of runs

    if do_splitted_max:
        times = np.array([tmax])
    else:
        times = [0.64, 2.0, 6.4, 20.0, 200.0]

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells14, cache, param, times, 
            confine_x=0.05, bin_count=40, histo_opts={'label': '${Tmax:.1f}$'}, 
            constant_particle_count=nparticles, additional_confine_ranges=[(1, 1, np.inf)])
    def mod_b(b):
        b.set(nthreads=1)
        b.cache = None
        b.cache_not_found_regenerate = False
    
    def mod_h(h):
        h.set(bin_width=0.005)

    def mod_h2(h):
        T = float(h.name.split('=')[1])
        h.set(normalize='density', manual_normalization_factor=T)

    histosetx.map_tree(mod_b, "batch")
    histosetx.map_tree(lambda h: mod_h(h) or mod_h2(h), "histo")
    histosetp.map_tree(lambda h : h.set(bin_width=1/60), "histo")
    powerlaw.set(plot_kwargs=powerlaw.get_kwargs()['plot_kwargs'] | {'color': 'k'})

    pointstmax = histosetx.search_parent('points_T=' + str(tmax))
    if not pointstmax is None:
        nsplit = 8
        splitted_points = chains.get_splitted_points(pointstmax, nsplit)
        
        if __name__ == '__main__' and do_splitted_max:
            multiproc.run_multiproc(splitted_points)
        
        mergedpoints = PointsMergeNode(pointstmax.name + "_merged", splitted_points, cache=cache, ignore_cache=False)

        valuesxtmax = histosetx.search_parent('valuesx_T=' + str(tmax))
        valuesptmax = histosetp.search_parent('valuesp_T=' + str(tmax))
        valuesxtmax.parents = {'points' : mergedpoints}
        valuesptmax.parents = {'points' : mergedpoints}

    N0_base = 0.0054 * 200
    dict2d = {
         # T : xmin, xmax, pmin, pmax, N0_corr
         #0.64 : (((-0.1, 0.05), (1, 3)), ),
         2.0 : (((-0.15, 0.1), (1, 10)), 1),
         6.4 : (((-0.2, 0.2), (1, 100)), 1),
         20 : (((-0.2, 0.5), (1, 300)), 1),
         200 : (((-0.2, 5), (1, 300)), 1),
    }

    topt_param = {
                   'x0': 0,
                   'y0': 1, 
                   'beta_s': param['beta_s'],
                   'q': param['q'],
                   'r': 4
                 }

    contourlevels = np.linspace(-6, 1, 20)

    nfmt = NodeFigureFormat(base=formats.doublehistand2dmany)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig[0].format(xlim=(-0.2, 0.6), ylim=(1e-2, 100))
    nfig[1].format(xlim=(1, 1000), ylim=(1e-5, 10))
    nfig[1].format(xformatter='log')

    for n, (T, (lims, N0_corr)) in enumerate(dict2d.items()):
        histo2d, toptygin = contourpl(T, lims, N0_corr, topt_param, N0_base, histosetx, contourlevels, cache)
        nfig[n + 2].format(xlim=lims[0], ylim=lims[1])
        nfig.add(histo2d, n + 2)
        nfig.add(toptygin, n + 2)

    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(powerlaw, 1)
    nfig[1].legend()
    #nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')


"""
= 14a retry
"""
def kruells14ax2a():
    name = inspect.currentframe().f_code.co_name
    kruells14ax2_blueprint(name, dict(dx_adv=0.00053, delta=0.282, sigma=1.07, beta_s=0.06, r=4, n_timesteps=20000))

def kruells14ax2b():
    name = inspect.currentframe().f_code.co_name
    kruells14ax2_blueprint(name, dict(dx_adv=0.00053, delta=0.282, sigma=0.95, beta_s=0.06, r=4, n_timesteps=20000))

"""
this one is fire
"""
def kruells14ax2c():
    name = inspect.currentframe().f_code.co_name
    kruells14ax2_blueprint(name, dict(dx_adv=0.00053, delta=0.323, sigma=0.95, beta_s=0.06, r=4, n_timesteps=20000))

def kruells14ax2d():
    name = inspect.currentframe().f_code.co_name
    kruells14ax2_blueprint(name, dict(dx_adv=0.00053, delta=0.323, sigma=1.07, beta_s=0.06, r=4, n_timesteps=20000))

def get_merged(histosetx, histosetp, T, nsplit, cache, start=0):
    """
    Replaces the parent points of histoset* by a merged point node
    containing the splitted nodes.
    Returns an array with the splitted point nodes

    can only be used once per T
    """

    pointsT = histosetx.search_parent('points_T=' + str(T))
    splitted_points = chains.get_splitted_points(pointsT, nsplit, start=start)
    mergedpoints = PointsMergeNode(pointsT.name + "_merged", splitted_points, cache=cache, ignore_cache=False)

    valuesxT = histosetx.search_parent('valuesx_T=' + str(T))
    valuespT = histosetp.search_parent('valuesp_T=' + str(T))
    valuesxT.parents = {'points' : mergedpoints}
    valuespT.parents = {'points' : mergedpoints}

    return splitted_points

""" same as 14ax2_blueprint but all points splitted and a little better management of the splitting """
""" WATCH OUT! the batch nodes aren't cached, neither are their kwargs. so they will not respond to parameter changes """
""" parameters should not be changed because splitted nodes are meant to accumulate over multiple runs """
""" for new parameters (above the splitting stage), copy, paste, and start with clean cache """
def kruells14ax3_blueprint(name, num_params):
    param = { 
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
            }

    #param_calc, _ = chains.param_from_numerical(dx_adv=0.00053, delta=0.282, sigma=1.07, beta_s=0.06, r=4, n_timesteps=20000)
    param_calc, _ = chains.param_from_numerical(**num_params)
    param |= param_calc
    print(param)

    #nparticles = 360000 # should be 8 hours..., used for first split round
    nparticles = 36000

    times = [0.64, 2.0, 6.4, 20.0, 200.0]
    bin_widths_x = {0.64: 0.0008,
                    2.0 : 0.001,
                    6.4: 0.001,
                    20.0: 0.003,
                    200.0: 0.008
                   }
    bin_widths_p = {0.64: 1/120,
                    2.0 : 1/60,
                    6.4: 1/45,
                    20.0: 1/30,
                    200.0: 1/10
                   }

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells14, cache, param, times, 
            confine_x=0.05, bin_count=40, histo_opts={'label': '${Tmax:.1f}$'}, 
            constant_particle_count=nparticles, additional_confine_ranges=[(1, 1, np.inf)])

    run_new_splits = False

    def mod_b(b):
        b.set(nthreads=1)
        b.cache = None
        b.cache_not_found_regenerate = run_new_splits
    
    def mod_h(h):
        T = float(h.name.split('=')[1])
        h.set(bin_width=bin_widths_x[T])
        h.set(normalize='density', manual_normalization_factor=T)

    def mod_p(h):
        T = float(h.name.split('=')[1])
        h.set(bin_width=bin_widths_p[T])

    histosetx.map_tree(mod_b, "batch")
    histosetx.map_tree(lambda h: mod_h(h), "histo")
    histosetp.map_tree(lambda h : mod_p(h), "histo")
    powerlaw.set(plot_kwargs=powerlaw.get_kwargs()['plot_kwargs'] | {'color': 'k'})

    # inject splitting behaviour
    # increase history by 1 to create a new round of splits. if doing so, 
    # run_new_splits further up must be set to True.
    # for loading stuff it must be set to False
    history = 12 # number of historical runs with splits that should be concatenated, 
    nsplit = {0.64: 1, 2.0: 1, 6.4: 1, 20.0: 1, 200.0: 8}
    splitted_points = []
    splitted_points_most_recent = []
    splitted_points_dict = {}

    # create newest split nodes
    for T, nsplit_ in nsplit.items():
        splitted_points_dict[T] = []
        pointsT = histosetx.search_parent('points_T=' + str(T))
        for h in range(history + 1):
            s_ = chains.get_splitted_points(pointsT, nsplit_, start=h * nsplit_)

            if h == history:
                splitted_points_most_recent += s_

            splitted_points += s_
            splitted_points_dict[T] += s_

        mergedpoints = PointsMergeNode(pointsT.name + "_merged", splitted_points_dict[T], cache=cache, ignore_cache=False)
        valuesxT = histosetx.search_parent('valuesx_T=' + str(T))
        valuespT = histosetp.search_parent('valuesp_T=' + str(T))
        valuesxT.parents = {'points' : mergedpoints}
        valuespT.parents = {'points' : mergedpoints}


    # run them (or load from cache if no new ones are created)
    if __name__ == '__main__' and run_new_splits:
        multiproc.run_multiproc(splitted_points_most_recent, exit_on_finish=True)

    # finished splitting, histosetx and histosetp should now be connected to parents that are merged point nodes containing all point nodes

    N0_base = 0.0054 * 200
    dict2d = {
         # T : xmin, xmax, pmin, pmax, N0_corr, cut_topt_levels
         #0.64 : (((-0.1, 0.05), (1, 3)), ),
         2.0 : (((-0.15, 0.1), (1, 10)), 1, 5),
         6.4 : (((-0.2, 0.2), (1, 100)), 1, 2),
         20 : (((-0.2, 0.5), (1, 300)), 1, 1),
         200 : (((-0.2, 3.5), (1, 300)), 1, None),
    }

    topt_param = {
                   'x0': 0,
                   'y0': 1, 
                   'beta_s': param['beta_s'],
                   'q': param['q'],
                   'r': 4
                 }

    contourlevels = np.linspace(-5, 1, 16)

    nfig = NodeFigure(formats.doublehist2)
    nfig.format(suptitle="Spatial and momentum spectrum of particles for pure diffusive shock acceleration")
    nfig[0].format(xlim=(-0.2, 0.6), ylim=(1e-2, 300))
    nfig[0].annotate('$\\delta =0.323,~~\\sigma=0.95$', (0.06, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
    nfig[1].format(xlim=(1, 3000), ylim=(3e-7, 10), xlabel="$p/p_\\mathrm{inj}$ at the shock")
    nfig[1].format(xformatter='log')
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(powerlaw, 1)
    #nfig[1].legend()
    handles_ts = []
    for n in histosetp.search_parents_all('histop'):
        handles_ts += n.handles
    nfig[1].legend(loc='ur', handles=handles_ts, ncols=1, title='Observation time $T$')
    nfig[1].legend(loc='ll', handles=powerlaw.handles,ncols=1, title='Powerlaw fit')
    nfig.savefig(figdir + '/' + name + '.pdf')

    nfigc = NodeFigure(formats.contours4)
    #nfigc.format(suptitle=)

    from matplotlib import lines

    for n, (T, (lims, N0_corr, cut_levels_topt)) in enumerate(dict2d.items()):
        histo2d, toptygin = contourpl(T, lims, N0_corr, topt_param, N0_base, histosetx, contourlevels, cache, labels_hist=False, cut_levels_topt=cut_levels_topt, topt_detail=200, bins=25)
        nfigc[n].format(xlim=lims[0], ylim=lims[1])
        nfigc.add(histo2d, n)
        nfigc.add(toptygin, n)
        #nfigc[n].format(title="Contour plot of $\\log{{\\bar F}}$ at $T={}$".format(T))
        nfigc[n].annotate("$T={}$".format(T), (0.06, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
        handles_contour = [lines.Line2D([], [], color='k', label='Number density')] + toptygin.handles
        hist2dhandle = histo2d.handles
        if n in [1, 3]:
            nfigc[n].format(ylabel="")
        if n in [0, 1]:
            nfigc[n].format(xlabel="")

    nfigc.figure.colorbar(hist2dhandle, loc='b')
    nfigc.figure.legend(loc='t', handles=handles_contour, ncols=2, title="Contour plots of $\\log{{\\bar F}}$")
    nfigc.savefig(figdir + '/' + name + '_contours.pdf')


def kruells14ax3c():
    name = inspect.currentframe().f_code.co_name
    kruells14ax3_blueprint(name, dict(dx_adv=0.00053, delta=0.323, sigma=0.95, beta_s=0.06, r=4, n_timesteps=20000))


cgs_gauss_unit = u.Unit("cm(-1/2) g(1/2) s-1")
cgs_gauss_eq = (u.G, cgs_gauss_unit, lambda x: x, lambda x: x)

def sigma_from_B(B):
    B = B.to(cgs_gauss_unit, equivalencies=[cgs_gauss_eq])
    sigma = 4 * constants.e.gauss**4 * B**2 / (9 * constants.m_e**4 * constants.c**6)
    return sigma.decompose()

def B_from_sigma(sigma):
    Bsq = sigma * (9 * constants.m_e**4 * constants.c**6) / (4 * constants.e.gauss**4)
    B = np.sqrt(Bsq)
    return B.decompose().to("G", equivalencies=[cgs_gauss_eq])

def sigma_bar_from_B(B, code_units):
    sigma = sigma_from_B(B)
    sigma_bar = code_units['kappa_0'] * code_units['p_inj'] * sigma / (constants.c**2 * code_units['beta_0']**2)
    return sigma_bar.decompose().value

def B_from_sigma_bar(sigma_bar, code_units):
    sigma = sigma_bar * (constants.c**2 * code_units['beta_0']**2) / (code_units['kappa_0'] * code_units['p_inj'])
    return B_from_sigma(sigma)

def ksyn_param_sets(param, ksyns, code_units=None):
    if code_units is None: 
        return {'k_syn={}'.format(k) : {'param' : param | {'k_syn': k}} for k in ksyns}
    else:
        return {'k_syn={}'.format(k) : {'param' : param | {'k_syn': k, 'B' : B_from_sigma_bar(k, code_units).value}} for k in ksyns}

""" same as 14ax3_blueprint but all points splitted and a little better management of the splitting """
""" but with weak synchrotron losses """
def kruells14ax4_blueprint(name, num_params):
    time = 50.0

    #nparticles = 360000 # should be 8 hours..., used for first split round
    nparticles = 36000 * 24

    #param_calc, _ = chains.param_from_numerical(dx_adv=0.00053, delta=0.282, sigma=1.07, beta_s=0.06, r=4, n_timesteps=20000)
    param_calc, _ = chains.param_from_numerical(**num_params) # old: 4, losses: 0.0005, 4: noloss: 0
    param_calc |= {'x0' : 0, 'y0' : 1, 't_inj' : time / nparticles} | param_calc | {'Tmax' : time}
    #param_sets = {'losses': {'param': {'k_syn' : 0.0005, 'x0' : 0, 'y0' : 1, 't_inj' : time / nparticles} | param_calc | {'Tmax' : time}},
    #        'noloss': {'param' : {'k_syn' : 0, 'x0' : 0, 'y0' : 1, 't_inj' : time / nparticles} | param_calc | {'Tmax' : time}},
    #        }
    ksyns = [0.0, 0.0005, 0.005, 0.05]
    param_sets = ksyn_param_sets(param_calc, ksyns)
    print(param_sets)

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp = chains.get_chain_parameter_series(PyBatchKruells14, cache, param_sets, 0.1, bin_count=40, histo_opts={'label': '${Tmax:.1f}$'}, additional_confine_ranges=[(1, 0, np.inf)])
    powerlaws = NodeGroup('plgroup', parents=[PowerlawNode('pl', {'dataset' : histosetp['k_syn=' + str(ksyn)]}, plot_kwargs={'color' : 'k'}) for ksyn in ksyns])

    run_new_splits = False

    def mod_b(b):
        b.set(nthreads=1)
        b.cache = None
        b.cache_not_found_regenerate = run_new_splits
    
    def mod_h(h):
        h.set(bin_width=0.005)

    def mod_h2(h):
        h.set(normalize='density', manual_normalization_factor=time)

    histosetx.map_tree(mod_b, "batch")
    histosetx.map_tree(lambda h: mod_h(h) or mod_h2(h), "histo")
    histosetp.map_tree(lambda h : h.set(bin_width=1/60), "histo")

    # inject splitting behaviour
    # increase history by 1 to create a new round of splits. if doing so, 
    # run_new_splits further up must be set to True.
    # for loading stuff it must be set to False
    #history = 0 # number of historical runs with splits that should be concatenated; now in the second tuple element of nsplit
    nsplit = {0.0 : (8, 0), 0.0005: (8, 0), 0.005: (8, 0), 0.05: (8,0)}
    splitted_points = []
    splitted_points_most_recent = []
    splitted_points_dict = {}

    # create newest split nodes
    for key, (nsplit_, history) in nsplit.items():
        synkey = 'k_syn=' + str(key)
        splitted_points_dict[synkey] = []
        pointskey = histosetx.search_parent('points_' + str(synkey))
        for h in range(history + 1):
            s_ = chains.get_splitted_points(pointskey, nsplit_, start=h * nsplit_)

            if h == history:
                splitted_points_most_recent += s_

            splitted_points += s_
            splitted_points_dict[synkey] += s_

        mergedpoints = PointsMergeNode(pointskey.name + "_merged", splitted_points_dict[synkey], cache=cache, ignore_cache=False)
        valuesxkey = histosetx.search_parent('valuesx_' + str(synkey))
        valuespkey = histosetp.search_parent('valuesp_' + str(synkey))
        valuesxkey.parents = {'points' : mergedpoints}
        valuespkey.parents = {'points' : mergedpoints}


    # run them (or load from cache if no new ones are created)
    if __name__ == '__main__' and run_new_splits:
        multiproc.run_multiproc(splitted_points_most_recent, exit_on_finish=True)

    # finished splitting, histosetx and histosetp should now be connected to parents that are merged point nodes containing all point nodes

    N0_base = 0.0054 * 200
    dict2d = {
         # T : xmin, xmax, pmin, pmax, N0_corr
         #0.64 : (((-0.1, 0.05), (1, 3)), ),
         2.0 : (((-0.15, 0.1), (1, 10)), 1),
         6.4 : (((-0.2, 0.2), (1, 100)), 1),
         20 : (((-0.2, 0.5), (1, 300)), 1),
         200 : (((-0.2, 5), (1, 300)), 1),
    }

    topt_param = {
                   'x0': 0,
                   'y0': 1, 
                   'beta_s': param_calc['beta_s'],
                   'q': param_calc['q'],
                   'r': 4
                 }

    contourlevels = np.linspace(-6, 1, 20)

    nfmt = NodeFigureFormat(base=formats.doublehistand2dmany)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig[0].format(xlim=(-0.2, 0.6), ylim=(1e-2, 100))
    nfig[1].format(xlim=(0.001, 1000), ylim=(1e-5, 10))
    nfig[1].format(xformatter='log')

    #for n, (T, (lims, N0_corr)) in enumerate(dict2d.items()):
    #    histo2d, toptygin = contourpl(T, lims, N0_corr, topt_param, N0_base, histosetx, contourlevels, cache)
    #    nfig[n + 2].format(xlim=lims[0], ylim=lims[1])
    #    nfig.add(histo2d, n + 2, instant=False)
    #    nfig.add(toptygin, n + 2, instant=False)

    nfig.add(histosetx, 0, instant=False)
    nfig.add(histosetp, 1, instant=False)
    nfig.add(powerlaws, 1, instant=False)
    #nfig.show_nodes('treetest.pdf')
    nfig[1].legend()
    #nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

def kruells14ax4c():
    name = inspect.currentframe().f_code.co_name
    kruells14ax4_blueprint(name, dict(dx_adv=0.00053, delta=0.323, sigma=0.95, beta_s=0.06, r=4, n_timesteps=20000))


""" same as 14ax3_blueprint but all points splitted and a little better management of the splitting """
""" but with strong synchrotron losses """
def kruells14ax5_blueprint(name, num_params):
    time = 50.0

    #nparticles = 360000 # should be 8 hours..., used for first split round
    nparticles = 36000 * 10

    #param_calc, _ = chains.param_from_numerical(dx_adv=0.00053, delta=0.282, sigma=1.07, beta_s=0.06, r=4, n_timesteps=20000)
    param_calc, _ = chains.param_from_numerical(**num_params)
    param_sets = {'losses': {'param': {'k_syn' : 0.005, 'x0' : 0, 'y0' : 1, 't_inj' : time / nparticles} | param_calc | {'Tmax' : time}},
            'stronglosses': {'param' : {'k_syn' : 0.05, 'x0' : 0, 'y0' : 1, 't_inj' : time / nparticles} | param_calc | {'Tmax' : time}},
            }
    print(param_sets)

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp = chains.get_chain_parameter_series(PyBatchKruells14, cache, param_sets, 0.05, bin_count=40, histo_opts={'label': '${Tmax:.1f}$'}, additional_confine_ranges=[(1, 0, np.inf)])
    powerlaws = NodeGroup('plgroup', parents=[PowerlawNode('pl', {'dataset' : histosetp['stronglosses']}, plot_kwargs={'color' : 'k'}),
                                              PowerlawNode('pl', {'dataset' : histosetp['losses']}, plot_kwargs={'color' : 'k'})
                                              ])

    run_new_splits = False

    def mod_b(b):
        b.set(nthreads=1)
        b.cache = None
        b.cache_not_found_regenerate = run_new_splits
    
    def mod_h(h):
        h.set(bin_width=0.005)

    def mod_h2(h):
        h.set(normalize='density', manual_normalization_factor=time)

    histosetx.map_tree(mod_b, "batch")
    histosetx.map_tree(lambda h: mod_h(h) or mod_h2(h), "histo")
    histosetp.map_tree(lambda h : h.set(bin_width=1/60), "histo")

    # inject splitting behaviour
    # increase history by 1 to create a new round of splits. if doing so, 
    # run_new_splits further up must be set to True.
    # for loading stuff it must be set to False
    #history = 0 # number of historical runs with splits that should be concatenated; now in the second tuple element of nsplit
    nsplit = {'stronglosses' : (8, 0), 'losses': (8, 0) }
    splitted_points = []
    splitted_points_most_recent = []
    splitted_points_dict = {}

    # create newest split nodes
    for key, (nsplit_, history) in nsplit.items():
        splitted_points_dict[key] = []
        pointskey = histosetx.search_parent('points_' + str(key))
        for h in range(history + 1):
            s_ = chains.get_splitted_points(pointskey, nsplit_, start=h * nsplit_)

            if h == history:
                splitted_points_most_recent += s_

            splitted_points += s_
            splitted_points_dict[key] += s_

        mergedpoints = PointsMergeNode(pointskey.name + "_merged", splitted_points_dict[key], cache=cache, ignore_cache=True)
        valuesxkey = histosetx.search_parent('valuesx_' + str(key))
        valuespkey = histosetp.search_parent('valuesp_' + str(key))
        valuesxkey.parents = {'points' : mergedpoints}
        valuespkey.parents = {'points' : mergedpoints}


    # run them (or load from cache if no new ones are created)
    if __name__ == '__main__' and run_new_splits:
        multiproc.run_multiproc(splitted_points_most_recent, exit_on_finish=True)

    # finished splitting, histosetx and histosetp should now be connected to parents that are merged point nodes containing all point nodes

    N0_base = 0.0054 * 200
    dict2d = {
         # T : xmin, xmax, pmin, pmax, N0_corr
         #0.64 : (((-0.1, 0.05), (1, 3)), ),
         2.0 : (((-0.15, 0.1), (1, 10)), 1),
         6.4 : (((-0.2, 0.2), (1, 100)), 1),
         20 : (((-0.2, 0.5), (1, 300)), 1),
         200 : (((-0.2, 5), (1, 300)), 1),
    }

    topt_param = {
                   'x0': 0,
                   'y0': 1, 
                   'beta_s': param_calc['beta_s'],
                   'q': param_calc['q'],
                   'r': 4
                 }

    contourlevels = np.linspace(-6, 1, 20)

    nfmt = NodeFigureFormat(base=formats.doublehistand2dmany)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig[0].format(xlim=(-0.2, 0.6), ylim=(1e-2, 100))
    #nfig[1].format(xlim=(1, 1000), ylim=(1e-5, 10))
    nfig[1].format(xformatter='log')

    #for n, (T, (lims, N0_corr)) in enumerate(dict2d.items()):
    #    histo2d, toptygin = contourpl(T, lims, N0_corr, topt_param, N0_base, histosetx, contourlevels, cache)
    #    nfig[n + 2].format(xlim=lims[0], ylim=lims[1])
    #    nfig.add(histo2d, n + 2, instant=False)
    #    nfig.add(toptygin, n + 2, instant=False)

    nfig.add(histosetx, 0, instant=False)
    nfig.add(histosetp, 1, instant=False)
    nfig.add(powerlaws, 1, instant=False)
    #nfig.show_nodes('treetest.pdf')
    nfig[1].legend()
    #nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')
    return
    param = { 
              'k_syn' : 0.005,
              'x0' : 0,
              'y0' : 1,
            }

    #param_calc, _ = chains.param_from_numerical(dx_adv=0.00053, delta=0.282, sigma=1.07, beta_s=0.06, r=4, n_timesteps=20000)
    param_calc, _ = chains.param_from_numerical(**num_params)
    param |= param_calc
    print(param)

    #nparticles = 360000 # should be 8 hours..., used for first split round
    nparticles = 36000 * 24 

    times = [50.0]

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells14, cache, param, times, 
            confine_x=0.05, bin_count=40, histo_opts={'label': '${Tmax:.1f}$'}, 
            constant_particle_count=nparticles, additional_confine_ranges=[(1, 1, np.inf)])

    run_new_splits = True

    def mod_b(b):
        b.set(nthreads=1)
        b.cache = None
        b.cache_not_found_regenerate = run_new_splits
    
    def mod_h(h):
        h.set(bin_width=0.005)

    def mod_h2(h):
        T = float(h.name.split('=')[1])
        h.set(normalize='density', manual_normalization_factor=T)

    histosetx.map_tree(mod_b, "batch")
    histosetx.map_tree(lambda h: mod_h(h) or mod_h2(h), "histo")
    histosetp.map_tree(lambda h : h.set(bin_width=1/60), "histo")
    powerlaw.set(plot_kwargs=powerlaw.get_kwargs()['plot_kwargs'] | {'color': 'k'})

    # inject splitting behaviour
    # increase history by 1 to create a new round of splits. if doing so, 
    # run_new_splits further up must be set to True.
    # for loading stuff it must be set to False
    history = 6 # number of historical runs with splits that should be concatenated, 
    nsplit = {50.0: 8}
    splitted_points = []
    splitted_points_most_recent = []
    splitted_points_dict = {}

    # create newest split nodes
    for T, nsplit_ in nsplit.items():
        splitted_points_dict[T] = []
        pointsT = histosetx.search_parent('points_T=' + str(T))
        for h in range(history + 1):
            s_ = chains.get_splitted_points(pointsT, nsplit_, start=h * nsplit_)

            if h == history:
                splitted_points_most_recent += s_

            splitted_points += s_
            splitted_points_dict[T] += s_

        mergedpoints = PointsMergeNode(pointsT.name + "_merged", splitted_points_dict[T], cache=cache, ignore_cache=True)
        valuesxT = histosetx.search_parent('valuesx_T=' + str(T))
        valuespT = histosetp.search_parent('valuesp_T=' + str(T))
        valuesxT.parents = {'points' : mergedpoints}
        valuespT.parents = {'points' : mergedpoints}


    # run them (or load from cache if no new ones are created)
    if __name__ == '__main__' and run_new_splits:
        multiproc.run_multiproc(splitted_points_most_recent, exit_on_finish=True)

    # finished splitting, histosetx and histosetp should now be connected to parents that are merged point nodes containing all point nodes

    N0_base = 0.0054 * 200
    dict2d = {
         # T : xmin, xmax, pmin, pmax, N0_corr
         #0.64 : (((-0.1, 0.05), (1, 3)), ),
         2.0 : (((-0.15, 0.1), (1, 10)), 1),
         6.4 : (((-0.2, 0.2), (1, 100)), 1),
         20 : (((-0.2, 0.5), (1, 300)), 1),
         200 : (((-0.2, 5), (1, 300)), 1),
    }

    topt_param = {
                   'x0': 0,
                   'y0': 1, 
                   'beta_s': param['beta_s'],
                   'q': param['q'],
                   'r': 4
                 }

    contourlevels = np.linspace(-6, 1, 20)

    nfmt = NodeFigureFormat(base=formats.doublehistand2dmany)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig[0].format(xlim=(-0.2, 0.6), ylim=(1e-2, 100))
    nfig[1].format(xlim=(1, 1000), ylim=(1e-5, 10))
    nfig[1].format(xformatter='log')

    for n, (T, (lims, N0_corr)) in enumerate(dict2d.items()):
        histo2d, toptygin = contourpl(T, lims, N0_corr, topt_param, N0_base, histosetx, contourlevels, cache)
        nfig[n + 2].format(xlim=lims[0], ylim=lims[1])
        nfig.add(histo2d, n + 2, instant=False)
        nfig.add(toptygin, n + 2, instant=False)

    nfig.add(histosetx, 0, instant=False)
    nfig.add(histosetp, 1, instant=False)
    nfig.add(powerlaw, 1, instant=False)
    #nfig.show_nodes('treetest.pdf')
    nfig[1].legend()
    #nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

def kruells14ax5c():
    name = inspect.currentframe().f_code.co_name
    kruells14ax5_blueprint(name, dict(dx_adv=0.00053, delta=0.323, sigma=0.95, beta_s=0.06, r=4, n_timesteps=20000))

""" same as 14ax4_blueprint but all points splitted and a little better management of the splitting """
""" but with SED """
def kruells14ay1_blueprint(name, num_params):
    time = 50.0

    #nparticles = 360000 # should be 8 hours..., used for first split round
    nparticles = 36000 * 24

    code_units = {'beta_0' : 0.01, 'kappa_0' : 1e21 * u.Unit("m2 s-1"), 'p_inj' : 1000 * constants.m_e * constants.c}

    #param_calc, _ = chains.param_from_numerical(dx_adv=0.00053, delta=0.282, sigma=1.07, beta_s=0.06, r=4, n_timesteps=20000)
    param_calc, _ = chains.param_from_numerical(**num_params) # old: 4, losses: 0.0005, 4: noloss: 0
    param_calc |= {'x0' : 0, 'y0' : 1, 't_inj' : time / nparticles} | param_calc | {'Tmax' : time}
    #param_sets = {'losses': {'param': {'k_syn' : 0.0005, 'x0' : 0, 'y0' : 1, 't_inj' : time / nparticles} | param_calc | {'Tmax' : time}},
    #        'noloss': {'param' : {'k_syn' : 0, 'x0' : 0, 'y0' : 1, 't_inj' : time / nparticles} | param_calc | {'Tmax' : time}},
    #        }
    ksyns = [0.0, 0.0005, 0.005]#, 0.05]
    param_sets = ksyn_param_sets(param_calc, ksyns, code_units)
    print(param_sets)

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp = chains.get_chain_parameter_series(PyBatchKruells14, cache, param_sets, 0.1, bin_count=40, histo_opts={'label': '$\\SI{{{B:.4f}}}{{G}}$'}, additional_confine_ranges=[(1, 0, np.inf)])
    powerlaws = NodeGroup('plgroup', parents=[PowerlawNode('pl', {'dataset' : histosetp['k_syn=' + str(ksyn)]}, plot_kwargs={'color' : 'k'}) for ksyn in ksyns])

    run_new_splits = False

    def mod_b(b):
        b.set(nthreads=1)
        b.cache = None
        b.cache_not_found_regenerate = run_new_splits
    
    def mod_h(h):
        h.set(bin_width=0.005)

    def mod_h2(h):
        h.set(normalize='density', manual_normalization_factor=time)

    histosetx.map_tree(mod_b, "batch")
    histosetx.map_tree(lambda h: mod_h(h) or mod_h2(h), "histo")
    histosetp.map_tree(lambda h : h.set(bin_width=1/30), "histo")

    # inject splitting behaviour
    # increase history by 1 to create a new round of splits. if doing so, 
    # run_new_splits further up must be set to True.
    # for loading stuff it must be set to False
    #history = 0 # number of historical runs with splits that should be concatenated; now in the second tuple element of nsplit
    nsplit = {0.0 : (8, 0), 0.0005: (8, 0), 0.005: (8, 0)}#, 0.05: (8,0)}
    splitted_points = []
    splitted_points_most_recent = []
    splitted_points_dict = {}

    # create newest split nodes
    for key, (nsplit_, history) in nsplit.items():
        synkey = 'k_syn=' + str(key)
        splitted_points_dict[synkey] = []
        pointskey = histosetx.search_parent('points_' + str(synkey))
        for h in range(history + 1):
            s_ = chains.get_splitted_points(pointskey, nsplit_, start=h * nsplit_)

            if h == history:
                splitted_points_most_recent += s_

            splitted_points += s_
            splitted_points_dict[synkey] += s_

        mergedpoints = PointsMergeNode(pointskey.name + "_merged", splitted_points_dict[synkey], cache=cache, ignore_cache=False)
        valuesxkey = histosetx.search_parent('valuesx_' + str(synkey))
        valuespkey = histosetp.search_parent('valuesp_' + str(synkey))
        valuesxkey.parents = {'points' : mergedpoints}
        valuespkey.parents = {'points' : mergedpoints}


    # run them (or load from cache if no new ones are created)
    if __name__ == '__main__' and run_new_splits:
        multiproc.run_multiproc(splitted_points_most_recent, exit_on_finish=True)

    # finished splitting, histosetx and histosetp should now be connected to parents that are merged point nodes containing all point nodes



    ### start SED stuff ###

    use_ksyn = 0.005
    histogramp = histosetp.search_tree("histop_k_syn=" + str(use_ksyn))


    nu_range = np.logspace(10, 26, 200) * u.Hz
    gamma_inj = 1000
    p_inj = gamma_inj * constants.m_e * constants.c
    d_L = 1e27 * u.cm
    gamma_integrate = np.logspace(1, 9, 20)
    model_params = dict(delta_D=10, z=Distance(d_L).z, d_L=d_L, R_b=1e12 * u.cm)

    print(B_from_sigma_bar(use_ksyn, code_units))
    def cb(model_params, batch_params):
        B = B_from_sigma_bar(use_ksyn, code_units)
        return model_params | {'B' : B}

    scaling_factor = 1e12

    radiation_params = dict(plot=True, model_params=model_params, model_params_callback=cb, nu_range=nu_range, gamma_integrate=gamma_integrate, factor=scaling_factor, cache=cache, ignore_cache=False)
    transform = MomentumCount('mc', histogramp, plot=False, cache=cache, p_inj=p_inj)
    synchrotronflux = SynchrotronExactAgnPy('synchro', {'N_data' : transform}, **radiation_params)
    #synchrotronflux_compare = SynchrotronExactAgnPyCompare('synchrocomp', powerlaw, gamma_inj=gamma_inj, plot_kwargs={'linestyle': 'dotted'}, **(radiation_params | {'ignore_cache' : False}))
    #sscflux_compare = SSCAgnPyCompare('ssccomp', powerlaw, gamma_inj=gamma_inj, plot_kwargs={'linestyle': 'dotted'}, **(radiation_params | {'ignore_cache' : False}))
    synchrotronfluxdelta = SynchrotronDeltaApproxAgnPy('synchrodelta', {'N_data' : transform}, plot_kwargs={'linestyle': 'dashed', 'alpha': 0.8, 'color'  : 'k'}, **radiation_params)
    sscflux = SynchrotronSelfComptonAgnPy('ssc', {'N_data' : transform}, plot_kwargs={'linestyle': 'dashdot'}, label='Delta approximation', **radiation_params)
    #synchropeak = VLineNode('synpeak', batch, callback=lambda p, c, **kw: c['label_fmt_fields']['B'])
    fluxes = NodeGroup('fluxgroup', [synchrotronflux, sscflux, synchrotronfluxdelta])#, sscflux_compare, synchrotronflux_compare])

    #histosetx = copy_to_group('groupx', histogramx, last_parents=points_range)
    #fluxset = copy_to_group('groupflux', fluxes, last_parents=points_range)
    #histosetp = NodeGroup('groupp', fluxset.search_parents_all('histop'))
    #powerlawset = NodeGroup('grouppl', fluxset.search_parents_all('powerlaw'))

    # single histogram figure
    nfig = NodeFigure(formats.singlehistSED, suptitle="Exemplary synthetic spectral energy distribution")
    nfig.add(histosetp, 0, plot_on="spectra")
    fluxsum = SynchrotronSum('synsum', [synchrotronflux, sscflux], label='Synchrotron + SSC', plot=True)#, plot_kwargs={'color': synchrotronfluxdelta.get_color()})#, sscflux_compare, synchrotronflux_compare]) 
    nfig.add(fluxsum, 1)
    nfig.add(synchrotronfluxdelta, 1)
    nfig[0].format(xlim=(0.8, 1e3), ylim=(1e-6, 2))
    nfig[0].annotate('$\\delta =0.323,~~\\sigma=0.95$', (0.06, 0.07), xycoords='axes fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
    nfig[1].format(ylim=(1e-16, 1e-9))
    pplt.rc['legend.fontsize'] = '8.0'
    nfig[1].legend(ncols=1, loc='uc', handles=fluxsum.handles + [
        Line2D([], [], label='Delta approximation', linestyle='dashed', color='k')])
        #Line2D([], [], linestyle='dashed', alpha=0.8, label='Synchrotron (delta approx.)', color='b'),
        #Line2D([], [], linestyle='dashdot', label='SSC', color='k', alpha=0.6)])
    nfig[0].legend(title='Magnetic field', loc='ur', ncols=1)
    nfig.savefig(figdir + '/' + name + '-single.pdf')

def kruells14ay1c():
    name = inspect.currentframe().f_code.co_name
    kruells14ay1_blueprint(name, dict(dx_adv=0.00053, delta=0.323, sigma=0.95, beta_s=0.06, r=4, n_timesteps=20000))

"""
using the numerical parameters
more detail
"""
def kruells14a1():
    name = inspect.currentframe().f_code.co_name

    param = { 
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
            }

    # judging from the dt study, 3000 timesteps should be enough
    # invdelta = 3.1
    param_calc, _ = chains.param_from_numerical(dx_adv=0.1, delta=0.323, sigma=0.95, beta_s=1, r=4, n_timesteps=5000)
    param |= param_calc
    print(param)

    if len(sys.argv) > 1:
        times = np.array([1130.0])
    else:
        times = np.array([3.6, 11.3, 36, 113, 1130])

    nparticles = 1e6

    cache = PickleNodeCache(cachedir, name)

    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells14, cache, param, times, confine_x=np.inf, bin_count=40, histo_opts={'label': '${Tmax:.2f}$'}, constant_particle_count=nparticles)
    def mod_b(b):
        b.set(nthreads=8)
        b.cache = None
        b.cache_not_found_regenerate = False#True
    
    def mod_h(h):
        h.set(bin_width=0.5)

    def mod_h2(h):
        T = float(h.name.split('=')[1])
        h.set(normalize='density', manual_normalization_factor=T)

    histosetx.map_tree(mod_b, "batch")
    histosetx.map_tree(lambda h: mod_h(h) or mod_h2(h), "histo")
    histosetp.map_tree(lambda h : h.set(bin_width=1/60), "histo")
    powerlaw.set(plot_kwargs=powerlaw.get_kwargs()['plot_kwargs'] | {'color': 'k'})
    t2d = 113
    valuesx = histosetx.search_parent('valuesx_T=' + str(t2d))
    valuesp = histosetp.search_parent('valuesp_T=' + str(t2d))

    tmax = 1130.0 # max(times)
    pointstmax = histosetx.search_parent('points_T=' + str(tmax))
    if not pointstmax is None:
        nsplit = 10
        splitted_points = chains.get_splitted_points(pointstmax, nsplit)
        
        if __name__ == '__main__' and len(sys.argv) > 1:
            multiproc.run_multiproc(splitted_points)
        
        mergedpoints = PointsMergeNode(pointstmax.name + "_merged", splitted_points, cache=cache, ignore_cache=False)

        valuesxtmax = histosetx.search_parent('valuesx_T=' + str(tmax))
        valuesptmax = histosetp.search_parent('valuesp_T=' + str(tmax))
        valuesxtmax.parents = {'points' : mergedpoints}
        valuesptmax.parents = {'points' : mergedpoints}

    contourlevels = np.linspace(-8, -1, 20)
    histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=25, log_bins=(False, True), plot=True, 
                              style='contour', normalize='density', log_histogram=True, ignore_cache=True,
                              cmap='Haline', plot_kwargs=dict(labels=True, levels=contourlevels, cmap_kw={'reverse': True}, robust=True, labels_kw={'color' : 'gray'}))

    topt_param = {
                   'x0': 0,
                   'y0': 1, 
                   'beta_s': 20, 
                   'q': 0.3,
                   'r': 4
                 }

    contour_xrange = (-40, 55)
    contour_yrange = (1, 500)
    toptygin = ToptyginContourNode('topt', plot=True, cache=cache,
                    params=topt_param, N0=0.2, T=6,
                    x_range=contour_xrange, y_range=(np.log10(contour_yrange[0]), np.log10(contour_yrange[1])), 
                    levels=contourlevels, detail=25,
                    contour_opts=dict(color='k', linestyle='--', linewidth=0.5, labels=False))

    nfmt = NodeFigureFormat(base=formats.doublehistand2d2)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig.format(suptitle="Spatial and momentum spectrum of particles for pure diffusive shock acceleration")
    nfig[2].format(title="Contour plot at $T={}$".format(t2d))
    nfig[0].format(xlim=(-30,100), ylim=(1e-4, 30))
    nfig[1].format(ylim=(1e-4, 1e1))
    nfig[2].format(xlim=contour_xrange, ylim=contour_yrange)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(powerlaw, 1)
    nfig.add(histo2d, 2)
    handles_ts = []
    for n in histosetp.search_parents_all('histop'):
        handles_ts += n.handles
    nfig[1].legend(loc='ur', handles=handles_ts, ncols=1, title='Observation time $T$')
    nfig[1].legend(loc='ll', handles=powerlaw.handles,ncols=1, title='Powerlaw fit')
    nfig.add(toptygin, 2)
    nfig.savefig(figdir + '/' + name + '.pdf')



"""
= 14 but some 2d stuff
parameter playground
"""
def kruells14a2():
    name = inspect.currentframe().f_code.co_name
    param = { 'Xsh' : 0.002,
              'beta_s' : 0.06,
              'r' : 4,
              'dt' : 0.001,
              't_inj' : 0.0003,
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
              'q' : 1,
              'Tmax' : 20.0
            }

    cache = PickleNodeCache(cachedir, name)
    batch = BatchNode('batch', batch_cls=PyBatchKruells14, cache=cache, param=param)
    points = PointNode('points', {'batch' : batch}, cache=cache)
    valuesx = ValuesNode('valuesx', {'points' : points}, index=0, cache=cache)
    valuesp = ValuesNode('valuesp', {'points' : points}, index=1, cache=cache)

    histox = HistogramNode('histox', {'values' : valuesx}, bin_count=40, log_bins=False, plot=True)
    histoy = HistogramNode('histoy', {'values' : valuesp}, bin_count=40, log_bins=True, plot=True)

    contourlevels = np.linspace(-4, 0, 12)
    histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=30, log_bins=(False, True), plot=True, 
                              style='contour', normalize='density', log_histogram=True, ignore_cache=True,
                              cmap='Haline', plot_kwargs=dict(labels=True, levels=contourlevels, cmap_kw={'reverse': True}, robust=True, labels_kw={'color' : 'gray'}))
    #histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=20, log_bins=(False, True), plot=True, style='contour', normalize='none', log_histogram=True, cmap='Boreal')

    s = 2
    u = 0.001
    topt_param = {
                   'x0': 0,
                   'y0': 1, 
                   'beta_s': 1 * s, 
                   'q': 4.24106 * u / s**2,
                   'r': 4
                 }

    print(cache)
    print(isinstance(cache, KwargsCacheMixin))
    toptygin = ToptyginContourNode('topt', plot=True, cache=cache,
                    params=topt_param, N0=0.2, T=1130,
                    x_range=(-0.4, 0.5), y_range=(0, np.log10(200)), 
                    levels=contourlevels, detail=60,
                    contour_opts=dict(color='k', linestyle='--', linewidth=0.5, labels=False))

    nfmt = NodeFigureFormat(base=formats.doublehistand2d)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig[2].format(xlim=(-0.4, 0.5), ylim=(1, 200))
    nfig.add(histox, 0)
    nfig.add(histoy, 1)
    nfig.add(histo2d, 2)
    nfig.add(toptygin, 2)
    #nfig.add(histosetp, 1)
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
using the numerical parameters
14b = testing some parameters
"""
def kruells14b1():
    name = inspect.currentframe().f_code.co_name

    param = { 
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
            }

    # judging from the dt study, 3000 timesteps should be enough
    # invdelta = 2.6
    param_calc, _ = chains.param_from_numerical(dx_adv=0.00005, delta=0.5, sigma=0.5, beta_s=1, r=4, n_timesteps=50000)
    param |= param_calc
    print(param)
    tmax = 1.25 #param['Tmax']

    do_splitted_max = False # if False do unsplitted rest of runs

    if do_splitted_max:
        times = np.array([tmax])
    else:
        times = tmax * np.array([0.001, 0.003, 0.007, 0.01, 0.03, 0.1, 1])

    nparticles = 4e5 # should be 4.5 hours if perfectly threaded

    cache = PickleNodeCache(cachedir, name)

    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells14, cache, param, times, confine_x=np.inf, bin_count=40, histo_opts={'label': '${Tmax:.4f}$'}, constant_particle_count=nparticles, additional_confine_ranges=[(1, 1, np.inf)])
    def mod_b(b):
        b.set(nthreads=1)
        b.cache = None
        b.cache_not_found_regenerate = True#do_splitted_max
    
    def mod_h(h):
        h.set(bin_width=0.002)

    def mod_h2(h):
        T = float(h.name.split('=')[1])
        h.set(normalize='density', manual_normalization_factor=T)
        #h.set(normalize='none')

    histosetx.map_tree(mod_b, "batch")
    histosetx.map_tree(lambda h: mod_h(h) or mod_h2(h), "histo")
    #histosetp.map_tree(lambda h : h.set(bin_width=1/60), "histo")
    powerlaw.set(plot_kwargs=powerlaw.get_kwargs()['plot_kwargs'] | {'color': 'k'})

    pointstmax = histosetx.search_parent('points_T=' + str(tmax))
    if not pointstmax is None:
        nsplit = 8
        splitted_points = chains.get_splitted_points(pointstmax, nsplit)
        
        if __name__ == '__main__' and do_splitted_max:
            multiproc.run_multiproc(splitted_points)
        
        mergedpoints = PointsMergeNode(pointstmax.name + "_merged", splitted_points, cache=cache, ignore_cache=False)

        valuesxtmax = histosetx.search_parent('valuesx_T=' + str(tmax))
        valuesptmax = histosetp.search_parent('valuesp_T=' + str(tmax))
        valuesxtmax.parents = {'points' : mergedpoints}
        valuesptmax.parents = {'points' : mergedpoints}

    nfmt = NodeFigureFormat(base=formats.doublehistand2d2)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig.format(suptitle="Spatial and momentum spectrum of particles for pure diffusive shock acceleration")
    #nfig[2].format(title="Contour plot at $T={}$".format(t2d))
    nfig[0].format(xlim=(-0.01, 0.07), ylim=(1e-4, 2))
    #nfig[1].format(ylim=(1e-4, 1e1))
    #nfig[2].format(xlim=contour_xrange, ylim=contour_yrange)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(powerlaw, 1)
    handles_ts = []
    for n in histosetp.search_parents_all('histop'):
        handles_ts += n.handles
    nfig[1].legend(loc='ur', handles=handles_ts, ncols=1, title='Observation time $T$')
    nfig[1].legend(loc='ll', handles=powerlaw.handles,ncols=1, title='Powerlaw fit')
    nfig.savefig(figdir + '/' + name + '.pdf')

"""
= 14 but some 2d stuff
with numerical comparison (wild)
and multiple time points
"""
#def kruells14a2():
#    name = inspect.currentframe().f_code.co_name
#    param = { 'Xsh' : 0.002,
#              'beta_s' : 0.06,
#              'r' : 4,
#              'dt' : 0.001,
#              't_inj' : 0.0003,
#              'k_syn' : 0,#.0001,
#              'x0' : 0,
#              'y0' : 1,
#              'q' : 1,
#              'Tmax' : 20.0
#            }
#
#    times = np.array([0.64, 2.0, 6.4, 20.0, 200.0])
#
#    cache = PickleNodeCache(cachedir, name)
#    batch = BatchNode('batch', batch_cls=PyBatchKruells14, cache=cache, param=param)
#    points = PointNode('points', {'batch' : batch}, cache=cache)
#    valuesx = ValuesNode('valuesx', {'points' : points}, index=0, cache=cache)
#    valuesp = ValuesNode('valuesp', {'points' : points}, index=1, cache=cache)
#
#    histox = HistogramNode('histox', {'values' : valuesx}, bin_count=40, log_bins=False, plot=True)
#    histoy = HistogramNode('histoy', {'values' : valuesp}, bin_count=40, log_bins=True, plot=True)
#
#generate_timerange_set(param, times, constant_particle_count=None):
#    points_range = {}
#    for name, kw in param_sets.items():
#        points_range[name] = {'points' : points.copy(name, last_kwargs=kw)}
#
#    valuesx = ValuesNode('valuesx', index=0, cache=cache, ignore_cache=False)
#    valuesp = ValuesNode('valuesp', index=1, cache=cache, ignore_cache=False,
#            confine_range=[(0, -confine_x, confine_x)],
#        )
#
#    histo_opts = {'bin_count' : bin_count, 'plot' : plot_on, 'cache' : cache, 'ignore_cache' : False} | histo_opts
#    histogramx = HistogramNode('histox', {'values' : valuesx}, log_bins=False if log_bins is None else log_bins[0], normalize='width', **histo_opts)
#    histogramp = HistogramNode('histop', {'values' : valuesp}, log_bins=True if log_bins is None else log_bins[1], normalize='density', **histo_opts)
#
#    histosetx = copy_to_group('groupx', histogramx, last_parents=points_range)
#
#    if powerlaws:
#        powerlaw = PowerlawNode('pl', {'dataset' : histogramp }, plot=plot_on)
#        powerlawset = copy_to_group('grouppl', powerlaw, last_parents=points_range)
#        return histosetx, powerlawset
#    else:
#        histosetp = copy_to_group('groupp', histogramp, last_parents=points_range)
#        return histosetx, histosetp
#
#def get_chain_times_maxpl(batch_cls, cache, param, times, confine_x=np.inf, bin_count=30, histo_opts={}, plot_on=True, log_bins=None, constant_particle_count=None):
#    param_sets = generate_timerange_set(param, times, constant_particle_count=constant_particle_count)
#    histosetx, histosetp = get_chain_parameter_series(batch_cls, cache, param_sets, confine_x, bin_count=bin_count, histo_opts=histo_opts, plot_on=plot_on, log_bins=log_bins)
#
#    max_histop = histosetp['T=' + str(max(times))]
#    powerlaw = PowerlawNode('pl', {'dataset' : max_histop}, plot=plot_on)#, color_cycle=cycle)
#
#    return histosetx, histosetp, powerlaw
#
#    contourlevels = np.linspace(-4, 0, 12)
#    histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=30, log_bins=(False, True), plot=True, 
#                              style='contour', normalize='density', log_histogram=True, ignore_cache=True,
#                              cmap='Haline', plot_kwargs=dict(labels=True, levels=contourlevels, cmap_kw={'reverse': True}, robust=True, labels_kw={'color' : 'gray'}))
#    #histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx, 'valuesy' : valuesp}, bin_count=20, log_bins=(False, True), plot=True, style='contour', normalize='none', log_histogram=True, cmap='Boreal')
#
#    topt_param = {
#                   'x0': 0,
#                   'y0': 1, 
#                   'beta_s': 0.2, 
#                   'q': 0.3,
#                   'r': 4
#                 }
#
#    print(cache)
#    print(isinstance(cache, KwargsCacheMixin))
#    toptygin = ToptyginContourNode('topt', plot=True, cache=cache,
#                    params=topt_param, N0=0.2, T=6,
#                    x_range=(-0.4, 0.5), y_range=(0, np.log10(200)), 
#                    levels=contourlevels, detail=60,
#                    contour_opts=dict(color='k', linestyle='--', linewidth=0.5, labels=False))
#
#    nfmt = NodeFigureFormat(base=formats.doublehistand2d)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
#    nfig = NodeFigure(nfmt)
#    nfig[2].format(xlim=(-0.4, 0.5), ylim=(1, 200))
#    nfig.add(histox, 0)
#    nfig.add(histoy, 1)
#    nfig.add(histo2d, 2)
#    nfig.add(toptygin, 2)
#    #nfig.add(histosetp, 1)
#    nfig.savefig(figdir + '/' + name + '.pdf')
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

    def k_syn_from_B(B):
        B_si = B.to("T", equivalencies=[gauss_tesla_eq])
        k_syn = B**2 / (6 * np.pi * constants.eps0) * (constants.e.si / (constants.m_e * constants.c))**4
        return (k_syn / u.Unit("kg-1 m-1")).decompose().value

    def B_from_k_syn(k_syn):
        k_syn = k_syn * u.Unit("kg-1 m-1")
        B = (constants.m_e * constants.c / constants.e.si)**2 * np.sqrt(6 * np.pi * constants.eps0 * k_syn)
        return B.to("G", equivalencies=[gauss_tesla_eq])

    ksyn = [0, 0.0001, 0.0005, k_syn_from_B(0.2 * u.G)]
    #Bs = [0 * u.G, 1e-8 * u.G]
    param_sets = {'k_syn={}'.format(ks) : {'param' : param | {'k_syn': ks}, 'label_fmt_fields' : {'B' : B_from_k_syn(ks).value }} for ks in ksyn}
    #param_sets = {'B={}'.format(B) : {'param' : param | {'k_syn': k_syn_from_B(B)}, 'label_fmt_fields' : {'B' : B}} for B in Bs}

    cache = PickleNodeCache(cachedir, name)
    batch = BatchNode('batch', batch_cls=PyBatchKruells9, cache=cache, ignore_cache=False)
    points = PointNode('points', {'batch' : batch}, cache=cache, ignore_cache=False)

    points_range = {}
    for n, kw in param_sets.items():
        points_range[n] = {'points' : points.copy(n, last_kwargs=kw)}

    valuesx = ValuesNode('valuesx', index=0, cache=cache, ignore_cache=False)
    valuesp = ValuesNode('valuesp', index=1, cache=cache, ignore_cache=False,
            confine_range=[(0, -50, 200)], # using all particles shifts the distribution to smaller momenta (probably through normalization)
            #confinements=[(0, lambda x: np.abs(x) <= 100)]
        )

    histo_opts = {'bin_count' : 50, 'plot' : 'spectra', 'cache' : cache, 'ignore_cache' : False, 'label' : '$k_\\mathrm{{syn}}=\\SI{{{B}}}{{\\gauss}}$'} 
    histogramx = HistogramNode('histox', {'values' : valuesx}, log_bins=False, normalize='width', **histo_opts)
    histogramp = HistogramNode('histop', {'values' : valuesp}, log_bins=True, normalize='density', **histo_opts)
    powerlaw = PowerlawNode('powerlaw', {'dataset' : LimitNode('limit', parents=histogramp, lower=1, upper=200)}, plot='spectra', errors=True, ignore_cache=False)
    
    nu_range = np.logspace(3, 19, 200) * u.Hz
    gamma_inj = 1000
    p_inj = gamma_inj * constants.m_e * constants.c
    d_L = 1e27 * u.cm
    gamma_integrate = np.logspace(1, 9, 20)
    model_params = dict(delta_D=10, z=Distance(d_L).z, d_L=d_L, R_b=1e16 * u.cm)

    def cb(model_params, batch_params):
        k_syn = batch_params['k_syn'] * u.Unit("kg-1 m-1")
        cgs_gauss_eq = (u.G, u.Unit("cm(-1/2) g(1/2) s-1"), lambda x: x, lambda x: x)
        B = (constants.m_e * constants.c / constants.e.si)**2 * np.sqrt(6 * np.pi * constants.eps0 * k_syn)
        B_cgs = (B * np.sqrt(4 * np.pi / constants.mu0)).to("gauss", equivalencies=[cgs_gauss_eq])
        print(B_cgs)
        return model_params | {'B' : B_cgs}

    print(param_sets)
    for p, ps in param_sets.items():
        print(cb({}, ps['param']))

    radiation_params = dict(plot=True, model_params=model_params, model_params_callback=cb, nu_range=nu_range, gamma_integrate=gamma_integrate, cache=cache, ignore_cache=False)
    transform = MomentumCount('mc', histogramp, plot=False, cache=cache, p_inj=p_inj)
    synchrotronflux = SynchrotronExactAgnPy('synchro', {'N_data' : transform}, **radiation_params)
    synchrotronflux_compare = SynchrotronExactAgnPyCompare('synchrocomp', powerlaw, gamma_inj=gamma_inj, plot_kwargs={'linestyle': 'dotted'}, **(radiation_params | {'ignore_cache' : False}))
    sscflux_compare = SSCAgnPyCompare('ssccomp', powerlaw, gamma_inj=gamma_inj, plot_kwargs={'linestyle': 'dotted'}, **(radiation_params | {'ignore_cache' : False}))
    synchrotronfluxdelta = SynchrotronDeltaApproxAgnPy('synchrodelta', {'N_data' : transform}, plot_kwargs={'linestyle': 'dashed', 'alpha': 0.6}, **radiation_params)
    sscflux = SynchrotronSelfComptonAgnPy('ssc', {'N_data' : transform}, plot_kwargs={'linestyle': 'dashdot'}, **radiation_params)
    #synchropeak = VLineNode('synpeak', batch, callback=lambda p, c, **kw: c['label_fmt_fields']['B'])
    fluxes = NodeGroup('fluxgroup', [synchrotronflux, sscflux, synchrotronfluxdelta, sscflux_compare, synchrotronflux_compare])

    histosetx = copy_to_group('groupx', histogramx, last_parents=points_range)
    fluxset = copy_to_group('groupflux', fluxes, last_parents=points_range)
    histosetp = NodeGroup('groupp', fluxset.search_parents_all('histop'))
    powerlawset = NodeGroup('grouppl', fluxset.search_parents_all('powerlaw'))

    nfig = NodeFigure(formats.doublehistSED)
    nfig.add(histosetx, 0, plot_on="spectra")
    nfig.add(powerlawset, 1, plot_on="spectra")
    nfig.add(fluxset, 2)
    #min_flux, max_flux = np.inf, 0
    #for fd in fluxset.search_parents_all("synchrodelta"):
    #    vals = fd.data[1][np.logical_and(fd.data[1] != 0, np.isfinite(fd.data[1])).nonzero()]
    #    if len(vals) > 0:
    #        min_flux = min(min(vals), min_flux)
    #        max_flux = max(max(vals), max_flux)
    #    
    #nfig[2].format(ylim=(min_flux.value, max_flux.value))
    nfig[2].format(ylim=(1e-31, 1e-18))
    pplt.rc['legend.fontsize'] = '8.0'
    nfig[2].legend(ncols=1, loc='uc', handles=[
        Line2D([], [], label='Synchrotron', color='k'),
        Line2D([], [], linestyle='dashed', alpha=0.6, label='Synchrotron (delta approx.)', color='k'),
        Line2D([], [], linestyle='dashdot', label='SSC', color='k'),
        Line2D([], [], linestyle='dotted', label='comparison (assuming perfect power laws)', color='k')])
    nfig.savefig(figdir + '/' + name + '.pdf')

    # single histogram figure
    nfig = NodeFigure(formats.singlehistSED)
    nfig.add(powerlawset, 0, plot_on="spectra")
    nfig.add(fluxset, 1)
    nfig[1].format(ylim=(1e-31, 1e-18))
    pplt.rc['legend.fontsize'] = '8.0'
    nfig[1].legend(ncols=1, loc='uc', handles=[
        Line2D([], [], label='Synchrotron', color='k'),
        Line2D([], [], linestyle='dashed', alpha=0.6, label='Synchrotron (delta approx.)', color='k'),
        Line2D([], [], linestyle='dashdot', label='SSC', color='k'),
        Line2D([], [], linestyle='dotted', label='comparison (assuming perfect power laws)', color='k')])
    nfig.savefig(figdir + '/' + name + '-single.pdf')

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
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruellsB1, cache, param, times, confine_x=5, bin_count=60)

    histosetp.map_tree(lambda h: h.set(label="${Tmax}$"), "histop")
    nfig = NodeFigure(formats.doublehist2, suptitle="Synchrotron losses without efficient acceleration")
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig[0].format(xlabel="$x$ (shock at $x=0$)", ylim=(1, 2e4))
    nfig[1].format(ylabel="", xlabel="$p/p_\\mathrm{inj}$ at the shock", ylim=(1e-4, 4))
    nfig[1].legend(loc='ll', title="Observation time $T$", ncols=1)
    #xlim=((None, None), (10**-1, 10**1)),
    #ylim=((None, None), (10**-4, 10**1)),
    nfig.savefig(figdir + '/' + name + '.pdf')

def kruellsB1c1():
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

    do_splitted_max = False # if False do unsplitted rest of runs

    tmax = times[-1]
    if do_splitted_max:
        times = [tmax]
    else:
        times = times#[:-1]

    nparticles = 1e4

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruellsB2, cache, param, times, confine_x=50, bin_count=100, constant_particle_count=nparticles)

    def mod_b(b):
        b.set(nthreads=1)
        b.cache = None
        b.cache_not_found_regenerate = do_splitted_max
    
    def mod_h(h):
        h.set(bin_width=0.5)

    def mod_h2(h):
        T = float(h.name.split('=')[1])
        h.set(normalize='density', manual_normalization_factor=T)

    histosetx.map_tree(mod_b, "batch")
    histosetx.map_tree(lambda h: mod_h(h) or mod_h2(h), "histo")
    histosetp.map_tree(lambda h : h.set(bin_width=1/60), "histo")
    powerlaw.set(plot_kwargs=powerlaw.get_kwargs()['plot_kwargs'] | {'color': 'k'})

    pointstmax = histosetx.search_parent('points_T=' + str(tmax))
    if not pointstmax is None:
        nsplit = 8
        splitted_points = chains.get_splitted_points(pointstmax, nsplit)
        
        if __name__ == '__main__' and do_splitted_max:
            multiproc.run_multiproc(splitted_points)
        
        mergedpoints = PointsMergeNode(pointstmax.name + "_merged", splitted_points, cache=cache, ignore_cache=False)

        valuesxtmax = histosetx.search_parent('valuesx_T=' + str(tmax))
        valuesptmax = histosetp.search_parent('valuesp_T=' + str(tmax))
        valuesxtmax.parents = {'points' : mergedpoints}
        valuesptmax.parents = {'points' : mergedpoints}

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
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruellsC1, cache, param, times, confine_x=0.2, bin_count=100)

    nfig = NodeFigure(formats.doublehist)
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    #xlim=((None, None), (10**-1, 10**2)),
    #ylim=((None, None), (10**-3, 10**0)),
    nfig.savefig(figdir + '/' + name + '.pdf')
    
def kruellsC1a1():
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

    times = np.array([20.0, 64.0, 640.0])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruellsC1, cache, param, times, confine_x=1, bin_count=100)
    histosetx.map_tree(lambda h: h.set(normalize='density'), 'histox')

    t2d = 640
    points2d = histosetx.search_parent('points_T=' + str(t2d))
    valuesx2d = ValuesNode('valuesx2d_T=' + str(t2d), parents={'points': points2d}, index=0, cache=cache)
    valuesp2d = ValuesNode('valuesp2d_T=' + str(t2d), parents={'points': points2d}, index=1, cache=cache)
    histosetp.map_tree(lambda n: n.set(label='${Tmax:.0f}$'), 'histop')

    contourlevels = np.linspace(-6, -1, 21)
    histo2d = Histogram2DNode('histo2d', {'valuesx' : valuesx2d, 'valuesy' : valuesp2d}, bin_count=25, log_bins=(False, True), plot=True, 
                              style='contour', normalize='density', log_histogram=True, ignore_cache=False, cache=cache, label='${Tmax:.1f}$',
                              cmap='Haline', plot_kwargs=dict(labels=True, levels=contourlevels, cmap_kw={'reverse': True}, robust=True, labels_kw={'color' : 'gray'}))

    nfmt = NodeFigureFormat(base=formats.doublehistand2d2)#,  axs_format={0: {'xlim' : (-0.8, 1)}})
    nfig = NodeFigure(nfmt)
    nfig.format(suptitle="Spatial and momentum spectrum of particles for pure stochastic acceleration")
    nfig[2].format(title="Contour plot of $\\log{{\\bar F}}$ for $T={}$".format(t2d))
    nfig[0].format(xlim=(-20, 20), ylim=(5e-4, 0.2))
    nfig[1].format(xlim=(0.2, 180), ylim=(1e-4, 2), xlabel="$p/p_\\mathrm{inj}$ at the shock")
    nfig[2].format(xlim=(-20, 20), ylim=(0.1, 200))
    #nfig[0].annotate('$\\delta =0.323,~~\\sigma=0.95$', (0.06, 0.92), xycoords='axes fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
    nfig.add(histosetx, 0)
    nfig.add(histosetp, 1)
    nfig.add(histo2d, 2)
    nfig[1].legend(title="Observation time $T$", ncols=1, loc='ur')
    #nfig.add(histo2d_2, 2)
    #handles_ts = []
    #for n in histosetp.search_parents_all('histop'):
    #    handles_ts += n.handles
    #nfig[1].legend(loc='ur', handles=handles_ts, ncols=1, title='Observation time $T$')
    #nfig[1].legend(loc='ll', handles=powerlaw.handles,ncols=1, title='Powerlaw fit')
    #handles_contour = []
    #for n in [histo2d, histo2d_2]:
    #    handles_contour += n.handles
    #nfig[2].legend(loc='ul', handles=handles_contour, ncols=1, title='Observation time $T$')
    nfig.savefig(figdir + '/' + name + '.pdf')

    
    
"""
C1a more particles x20
bullshit this are fewer particles
"""
def kruellsC1a2():
    name = inspect.currentframe().f_code.co_name
    param = { 'kappa' : 1,
              'a2' : 0.01,
              'k_syn' : 0,
              'dt' : 0.004,
              't_inj' : 0.00002,
              'x0' : 0,
              'y0' : 1,
              'Lx' : 20,
              'Lylower' : 0,
              'Lyupper' : 200,
            }

    times = np.array([64, 640])

    cache = PickleNodeCache(cachedir, name)
    histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruellsC1, cache, param, times, confine_x=0.2, bin_count=200, constant_particle_count=3e5)

    def mod_b(b):
        b.set(nthreads=1)
        b.cache = None
        b.cache_not_found_regenerate = True

    histosetx.map_tree(mod_b, "batch")

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
    if len(sys.argv) >= 3:
        logging.info("Loading kwargs")
        if sys.argv[1] == '-p':
            import pickle
            with open(sys.argv[2], mode='rb') as f:
                print(pickle.load(f))
        else:
            cache = PickleNodeCache('pickle', sys.argv[1])
            oldkw = cache.load_kwargs(sys.argv[2]) 
            print("old kwargs: ", oldkw)
        if len(sys.argv) == 5:
            oldkw[sys.argv[3]] = eval(sys.argv[4])
            print("overwrite with those kwargs?", oldkw)
            if input("Y/N") == "Y":
                cache.store_kwargs(sys.argv[2], oldkw) 
        exit()

    if len(sys.argv) == 2:
        eval("kruells" + sys.argv[1] + "()")
        exit()

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
    #kruells9b1()
    #kruells9a()
    #kruells9a1()
    #kruells9a2()
    #kruells9a3()
    #kruells14a()
    kruells14a1()
    #kruells14b1()
    #kruells14()

    #kruellsB1()
    #kruellsB1a()
    #kruellsB1b()
    #kruellsB1c()

    #kruellsC1a()
    #kruells12ts()
    #kruells12()
    #kruells13ts()

    #achterberg2()

