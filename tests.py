import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import time
import argparse

import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.special.kruells import *
from pybatch.special.sourcetest import *
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
""" Source Test """

@cached(**cache_opts)
def ex_sourcetest():
    param = { 'Tmax' : 1,
              'x_min' : -1,
              'x_max' : 1,
              'x0' : 0,
              'N' : 500000
            }

    exset = Experiment(PyBatchSourcetest, param)
    return exset.run()

test_sourcetest = ExporterHist(
        ex_sourcetest,
        store_opts,
        log_x=False,
        log_y=False,
        use_integrator=0,
        bin_count=200
    )


""" *********** """
""" Kruells 1 """

@cached(**cache_opts)
def ex_kruells1():
    param = { 'Tesc' : 0.25,
              'x0' : 0,
              'p0' : 0,
              'N' : 50
            }

    times = np.array([64, 200, 640, 1000]) / 100

    exset = ExperimentSet(PyBatchKruells1, generate_timerange(param, times))
    return exset.run()

test_kruells1 = ExporterDoubleHist(ex_kruells1, store_opts, log_x=(False, False), log_y=(False, False))


""" *********** """
""" Kruells 2 """

@cached(**cache_opts)
def ex_kruells2():
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

    exset = ExperimentSet(PyBatchKruells2, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells2 = ExporterDoubleHistPL(ex_kruells2, store_opts, powerlaw_annotate=True, log_x=(False, False), log_y=(True, True), bin_count=50)

""" *********** """
""" Kruells 2a (synchrotron) """

@cached(**cache_opts)
def ex_kruells2a():
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

    exset = ExperimentSet(PyBatchKruells2, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells2a = ExporterDoubleHist(ex_kruells2a, store_opts, log_x=(False, False), log_y=(True, True), bin_count=50)

""" *********** """
""" Kruells 3 """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection

    b: wider timerange, powerlaws
"""

@cached(**cache_opts)
def ex_kruells3():
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

    exset = ExperimentSet(PyBatchKruells3, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells3 = ExporterDoubleHist(
        ex_kruells3,
        store_opts,
        log_y=(True, True),
        #average_bin_size=200,
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "p/p_inj"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, True),
        powerlaw_annotate=True,
        xlim=((None, None), (None, None))
)

""" *********** """
""" Kruells 3a """
"""
    Synchrotron

"""

@cached(**cache_opts)
def ex_kruells3a():
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

    exset = ExperimentSet(PyBatchKruells3, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells3a = ExporterDoubleHist(
        ex_kruells3a,
        store_opts,
        log_y=(True, True),
        #average_bin_size=200,
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "p/p_inj"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, True),
        powerlaw_annotate=True,
        xlim=((None, None), (None, None))
)


""" *********** """
""" Kruells 4 """

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

@cached(**cache_opts)
def ex_kruells5():
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

    exset = ExperimentSet(PyBatchKruells5, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells5 = ExporterDoubleHistPL(
        ex_kruells5,
        store_opts,
        log_y=(True, True),
        #average_bin_size=200,
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "p/p_inj"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, True),
        powerlaw_annotate=True,
        xlim=((None, None), (None, None))
)


""" *********** """
""" Kruells 6a """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection

    parametrisation of 1992 paper

    synchrotron losses
"""

@cached(**cache_opts)
def ex_kruells6a():
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

    exset = ExperimentSet(PyBatchKruells6, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells6a = ExporterDoubleHist(
        ex_kruells6a,
        store_opts,
        log_y=(True, True),
        log_x=(False, False),
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "ln(p/p_inj)"),
        powerlaw_annotate=True,
        ylabels=("Particle count", "Particle count")
)




""" *********** """
""" Kruells 6b """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection

    parametrisation of 1992 paper

    b: wider timerange, powerlaws
"""

@cached(**cache_opts)
def ex_kruells6b():
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

    exset = ExperimentSet(PyBatchKruells6, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells6b = ExporterDoubleHistPL(
        ex_kruells6b,
        store_opts,
        log_y=(True, True),
        #average_bin_size=200,
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "ln(p/p_inj)"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, False),
        powerlaw_annotate=True,
        xlim=((None, None), (None, None))
)



""" *********** """
""" Kruells 6c """
"""
    u0 = 0
"""

@cached(**cache_opts)
def ex_kruells6c():
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

    exset = ExperimentSet(PyBatchKruells6, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells6c = ExporterDoubleHistPL(
        ex_kruells6c,
        store_opts,
        log_y=(True, True),
        #average_bin_size=200,
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "ln(p/p_inj)"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, False),
        powerlaw_annotate=True,
        xlim=((None, None), (None, None))
)

""" *********** """
""" Kruells 7 """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection
    (like kruells 923)
    but with a spatially variable diffusion coefficient

    parametrisation of 1992 paper

"""

@cached(**cache_opts)
def ex_kruells7():
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

    exset = ExperimentSet(PyBatchKruells7, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells7 = ExporterDoubleHistPL(
        ex_kruells7,
        store_opts,
        log_y=(True, True),
        #average_bin_size=200,
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "ln(p/p_inj)"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, False),
        powerlaw_annotate=True,
        xlim=((None, None), (None, None))
)

@cached(**cache_opts)
def ex_kruells7a():
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

    exset = ExperimentSet(PyBatchKruells7, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells7a = ExporterDoubleHistPL(
        ex_kruells7a,
        store_opts,
        log_y=(True, True),
        #average_bin_size=200,
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "ln(p/p_inj)"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, False),
        powerlaw_annotate=True,
        xlim=((None, None), (None, None))
)

@cached(**cache_opts)
def ex_kruells7b():
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

    exset = ExperimentSet(PyBatchKruells7, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells7b = ExporterDoubleHistPL(
        ex_kruells7b,
        store_opts,
        log_y=(True, True),
        #average_bin_size=200,
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "ln(p/p_inj)"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, False),
        powerlaw_annotate=True,
        xlim=((None, None), (None, None))
)

# this one is for the talk at FRANCI 21
@cached(**cache_opts)
def ex_kruells7c():
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

    exset = ExperimentSet(PyBatchKruells7, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells7c = ExporterDoubleHistConfineP(
        ex_kruells7c,
        store_opts,
        log_y=(True, True),
        x_range_for_p=5,
        #average_bin_size=200,
        bin_count=30,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "ln(p/p_inj)"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, False),
        title="Reproduction of Krülls (1994)",
        xlim=((None, None), (None, None))
)


# this one is for the talk at FRANCI 21 too
@cached(**cache_opts)
def ex_kruells7d():
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

    exset = ExperimentSet(PyBatchKruells7, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells7d = ExporterDoubleHistConfineP(
        ex_kruells7d,
        store_opts,
        log_y=(True, True),
        x_range_for_p=1,
        #average_bin_size=200,
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "ln(p/p_inj)"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, False),
        xlim=((None, None), (None, None))
)

# this one is for the talk at FRANCI 21 too
@cached(**cache_opts)
def ex_kruells7e():
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

    exset = ExperimentSet(PyBatchKruells7, generate_timerange(param, times))
    exset.run()

    return exset

test_kruells7e = ExporterDoubleHistConfineP(
        ex_kruells7e,
        store_opts,
        log_y=(True, True),
        x_range_for_p=1,
        #average_bin_size=200,
        bin_count=50,
        subtitles=("Spatial", "Momentum"),
        xlabels=("x", "ln(p/p_inj)"),
        ylabels=("Particle count", "Particle count"),
        log_x=(False, False),
        xlim=((None, None), (None, None))
)

""" *********** """
""" Kruells 8 """
""" Reproduce 1994/Fig. 2 and 3
    using continous pseudo particle injection
    (like kruells 923)
    but with a spatially variable diffusion coefficient

    parametrisation of 1992 paper

"""

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

""" ***************** """
""" Run experiment(s) """
if __name__ == '__main__':
    #test_kruells6b()
    #test_kruells3()
    #test_kruells3a()
    #test_kruells2()
    #test_kruells2a()
    #test_kruells5()
    #test_kruells8()
    #test_kruells4()
    test_kruells7c()
    #test_kruells7d()
    #test_kruells7e()
    #test_kruells7a()
    #test_kruells7b()
    #test_kruells6a()
    #test_kruells6c()