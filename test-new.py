import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from node.special import *
from node.node import *
from node.cache import PickleNodeCache
from pybatch.special.kruells import *
import proplot as pplt
import logging
import numpy as np
from node.nodefigure import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def kwargs_dict(**kwargs):
        return kwargs

def generate_timerange_set(param, times):
        l = []
        for t in times:
                l.append({'param': param | {'Tmax' : t}})

        return l

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
times = np.array([0.64, 2.0, 6.4, 20, 64])

cache = PickleNodeCache('testcache', '9a1')
batch = BatchNode('batch',
        batch_cls = PyBatchKruells9,
        cache=cache,
        ignore_cache=False
    )
points = PointNode('points', {'batch' : batch}, cache=cache, ignore_cache=False)
valuesx = ValuesNode('valuesx', {'points' : points}, index=0, cache=cache, ignore_cache=False)
valuesp = ValuesNode('valuesp', {'points' : points}, index=1, cache=cache, ignore_cache=True,
                confinements=[(0, lambda x : np.abs(x) < 0.01)],
        )

histo_opts = kwargs_dict(bin_count=50, plot=True, cache=cache, ignore_cache=True)

histogramx = HistogramNode('histox', {'values' : valuesx}, log_bins=False, normalize='width', **histo_opts)
histogramp = HistogramNode('histop', {'values' : valuesp}, log_bins=True, normalize='density', **histo_opts)

tr = generate_timerange_set(param, times)

histosetx = NodeSet('set', {'histo' : histogramx}, kwargs=tr)
histosetp = NodeSet('set', {'histo' : histogramp}, kwargs=tr)

powerlaw = PowerlawNode(
                'pl', 
                {'dataset' : NodeSetItem(
                        'item', 
                        {'set' : histosetp},
                        key=np.argmax(times))
                },
                errors=False,
                plot=True)

ff = NodeFigureFormat(
                subplots={'ncols': 2},
                fig_format={'yscale': 'log', 'ylabel':'particle number density', 'xformatter': pplt.SciFormatter(), 'yformatter': pplt.SciFormatter()},
                axs_format=[
                        {'xlabel': '$x$'},
                        {'xscale': 'log', 'xlabel': '$p/p_\\textrm{inj}$'}
                ],
                legends_kw={1: {'loc': 'ur', 'ncols': 1}}
        )

print(repr(ff))
nfig = NodeFigure(ff)
nfig.add(histosetx, 0)
nfig.add(powerlaw, 1)
nfig.savefig('test3.pdf')
