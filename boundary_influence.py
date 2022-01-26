import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.special.kruells import *
import proplot as pplt
import logging
import chains
import formats
from node.cache import PickleNodeCache

from powerlawseries import *

pplt.rc.update({
        'text.usetex' : True,
        })

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def cb(this_param, param):
        d = param | this_param
        return {'param': d, 'label_fmt_fields': d}

param =  {
        'r' : 4,
        't_inj' : 0.0005,
        'x0' : 0,
        'y0' : 1,
        'k_syn' : 0,
        'Tmax' : 300,
        'dt':  0.05,
        'Xsh' : 0.25,
        'beta_s' : 0.57,
        'q' : 5
}


name = 'boundary_11'
cache = PickleNodeCache('pickle', name)
chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells11, cache, 2, lambda c: c['batch_param']['L'])

var = PowerlawSeriesVariable('L', 'L', [3, 5, 6, 8, 10, 20, 30, 50, 80, 200, 1000])
pls = PowerlawSeries(chain, var, cb, callback_kwargs={'param': param})
pls.plot_datarow("figures/{}.pdf".format(name), formats.powerlaws, xlabel='$L$', xscale='log')
pls.plot_histograms("figures/{}_histograms.pdf".format(name), formats.doublehist)
