import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from node.special import *
from node.node import *
from node.graph import *
from node.cache import PickleNodeCache
from pybatch.special.kruells import *
import proplot as pplt
import logging
import numpy as np
from node.nodefigure import *
import formats
import chains

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def generate_timerange_set(param, times):
        l = {}
        for t in times:
                l[str(t)] = {'param': param | {'Tmax' : t}}

        return l

param = { 'Xsh' : 0.0028,
          'beta_s' : 0.06,
          'r' : 4,
          'dt' : 0.01,#0.001,
          't_inj' : 0.2,#0.0002,
          'k_syn' : 0,#.0001,
          'x0' : 0,
          'y0' : 1,
          'q' : 4/9,
        }
times = np.array([0.64, 2.0, 6.4, 20, 64])

cache = PickleNodeCache('testcache', '9a1')
tr = generate_timerange_set(param, times)
histosetx, histosetp = chains.get_chain_parameter_series(PyBatchKruells9, cache, tr, 0.01)

powerlaw = PowerlawNode(
                'pl', 
                {'dataset' : histosetp[str(max(times))]},
                errors=False,
                plot=True)

nfig = NodeFigure(formats.doublehist)
nfig.add(histosetx, 0)
nfig.add(powerlaw, 1)
nfig.savefig('test3.pdf')
nfig.show_nodes('test3nodes.pdf')

