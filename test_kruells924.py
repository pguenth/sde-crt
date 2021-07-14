import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import time
import argparse

from scipy.stats import linregress

import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.special.kruells92 import *
from pybatch.pybreakpointstate import *

from evaluation.experiment import *
from evaluation.helpers import *
from evaluation.extractors import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


plt_dir = "out"
plt_format = "pdf"
#matplotlib.use('GTK3Agg')



parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cache')
parser.add_argument('-r', '--regenerate', action='store_true')
args = parser.parse_args()
logging.info("Arguments: " + str(args))

param = { 'dxs' : 0.25,
          'Kpar' : 5,
          'Vs' : 1,
          'r' : 4,
          'dt' : 0.1,
          'r_inj' : 0.01,
          'beta_s' : 0,#.0001,
          'dx_inj' : 1,
          'N' : 50,
          'x0' : 0,
          'p0' : 0
        }

extractor_x = HistogramExtractorFinish(0)
extractor_p = HistogramExtractorFinish(1)

indizes = [0, 1]
times = np.array([64, 200, 640, 1000])

def gen():
    exset = ExperimentSet(PyBatchKruells924, generate_timerange(param, times))
    exset.run()
    return exset

exset = pickle_cache(args.cache, gen, args.regenerate)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
exset.plot(axs[0], extractor_x)
exset.plot(axs[1], extractor_p)
fig.savefig(plt_dir + "/timerange-both." + plt_format)

