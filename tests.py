import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import time
import argparse

import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.special.kruells92 import *
from pybatch.pybreakpointstate import *

from evaluation.experiment import *
from evaluation.helpers import *
from evaluation.extractors import *
from evaluation.exporters import *



""" ************************************************** """
""" Parsing of cmd line arguments and setting defaults """

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--autocache', action='store_true')
parser.add_argument('-c', '--cache')
parser.add_argument('-r', '--regenerate', action='store_true')
args = parser.parse_args()
logging.info("Arguments: " + str(args))

cache_opts = {
    'cachedir' : 'pickle',
    'filename' : args.cache,
    'regenerate' : args.regenerate
}
if args.autocache:
    del cache_opts['filename']

store_opts = {
    'dir' : 'out',
    'format' : 'pdf'
}


    

""" *********** """
""" Kruells 921 """

@cached(**cache_opts)
def ex_kruells921():
    param = { 'Tesc' : 0.25,
              'x0' : 0,
              'p0' : 0,
              'N' : 50
            }

    times = np.array([64, 200, 640, 1000]) / 100

    exset = ExperimentSet(PyBatchKruells921, generate_timerange(param, times))
    return exset.run()

test_kruells921 = ExporterDoubleHist(ex_kruells921, store_opts, log_x=(False, False), log_y=(False, False))



""" *********** """
""" Kruells 922 """

@cached(**cache_opts)
def ex_kruells922():
    param = { 'dxs' : 0.25,
              'Kpar' : 5,
              'Vs' : 1,
              'r' : 4,
              'dt' : 0.1,
              'r_inj' : 0.1,
              'beta_s' : 0,#.0001,
              'dx_inj' : 1,
              'x0' : 0,
              'p0' : 0
            }

    times = np.array([64, 200, 640, 1000])

    exset = ExperimentSet(PyBatchKruells922, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells922 = ExporterDoubleHist(ex_kruells922, store_opts, log_x=(False, False), log_y=(False, False))



""" *********** """
""" Kruells 923a """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection

    Working example, but r_inj << dt ?
"""

@cached(**cache_opts)
def ex_kruells923a():
    param = { 'dxs' : 0.25,
              'Kpar' : 5,
              'Vs' : 1,
              'r' : 4,
              'dt' : 0.1,
              'r_inj' : 0.005,
              'beta_s' : 0,#.0001,
              'x0' : 0,
              'p0' : 0
            }

    times = np.array([64, 200, 640, 1000, 2000])

    exset = ExperimentSet(PyBatchKruells923, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells923a = ExporterDoubleHist(
        ex_kruells923a,
        store_opts,
        log_y=(True, True),
        bin_count=100,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "ln(p/p_inj)"),
        ylabels=("Particle count", "Particle count")
)

""" *********** """
""" Kruells 923b """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection

    b: wider timerange
"""

@cached(**cache_opts)
def ex_kruells923b():
    param = { 'dxs' : 0.25,
              'Kpar' : 5,
              'Vs' : 1,
              'r' : 4,
              'dt' : 0.1,
              'r_inj' : 0.003,
              'beta_s' : 0,#.0001,
              'x0' : 0,
              'p0' : 0
            }

    times = np.array([20, 64, 200, 640, 1000, 2000])

    exset = ExperimentSet(PyBatchKruells923, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells923b = ExporterDoubleHist(
        ex_kruells923b,
        store_opts,
        log_y=(True, True),
        bin_count=100,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "ln(p/p_inj)"),
        ylabels=("Particle count", "Particle count")
)



""" *********** """
""" Kruells 924 """

@cached(**cache_opts)
def ex_kruells924():
    param = { 'dxs' : 0.25,
              'Kpar' : 5,
              'Vs' : 1,
              'r' : 4,
              'dt' : 0.1,
              'r_inj' : 0.1,
              'beta_s' : 0,#.0001,
              'dx_inj' : 1,
              'N' : 500,
              'x0' : 0,
              'p0' : 0
            }

    times = np.array([64, 200, 640, 1000])

    exset = ExperimentSet(PyBatchKruells924, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells924 = ExporterDoubleHist(ex_kruells924, store_opts, log_x=(False, False), log_y=(False, False))


""" ***************** """
""" Run experiment(s) """
if __name__ == '__main__':
    test_kruells923b()
