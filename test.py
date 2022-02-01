import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from node.cache import PickleNodeCache
from pybatch.special.kruells import *
import logging
from node.nodefigure import *
import formats
import chains

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

param = { 'Xsh' : 0.0028,
          'beta_s' : 0.06,
          'r' : 4,
          'dt' : 0.001,
          't_inj' : 0.0002,
          'k_syn' : 0,#.0001,
          'x0' : 0,
          'y0' : 1,
          'q' : 4/9,
        }
times = [0.64, 2.0, 6.4, 20, 64]

name = '9a1'
cache = PickleNodeCache('testcache', name)
histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(PyBatchKruells9, cache, param, times, 0.01)

nfig = NodeFigure(formats.doublehist)
nfig.add(histosetx, 0)
nfig.add(histosetp, 1)
nfig.add(powerlaw, 1)
nfig.savefig(name + '.pdf')

