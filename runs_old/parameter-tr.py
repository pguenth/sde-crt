import sys
sys.path.insert(0, 'lib')
from pybatch.special.kruells import *
import proplot as pplt
import logging
import chains
import formats
from grapheval.cache import PickleNodeCache
from grapheval.graph import draw_node_chain

from powerlawseries import *

pplt.rc.update({
                'text.usetex' : True,
                })

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

default_params = {
          'Xsh' : 0.002,
          'beta_s' : 0.08,
          'r' : 4,
          'dt' : 0.001,
          't_inj' : 0.00022, # -> sollten ca, 8 std sein
          'k_syn' : 0,
          'x0' : 0,
          'y0' : 1,
          'q' : 2,
        }

params = {
    'beta_s' : PowerlawSeriesVariable('\\beta_s', 'beta_s', [0.01, 0.06, 0.1, 0.6]),
    'q' : PowerlawSeriesVariable('q', 'q', [0.2, 1.5, 3, 20]),
    'Xsh' : PowerlawSeriesVariable('X_\\textrm{sh}', 'Xsh', [0.0001, 0.001, 0.005, 0.05]),
    't_inj' : PowerlawSeriesVariable('t_\\textrm{inj}', 't_inj', [0.00042, 0.0017]),
    'dt' : PowerlawSeriesVariable('\\Delta\\tau', 'dt', [0.0005, 0.0008, 0.0012, 0.002]),
    'r' : PowerlawSeriesVariable('r', 'r', [1.2, 2, 3.5, 5.5])
}

def cb(this_param, param):
    return {'param': param | this_param}

times = [0.64, 2.0, 6.4, 20, 200]
name = 'param_9a'

for var_name, var in params.items():
    x_callback = lambda common: common['batch_param'][var_name]
    this_name = '{}_{}'.format(name, var_name)
    cache = PickleNodeCache('pickle', this_name)
    chain = chains.get_chain_powerlaw_datapoint_tseries(PyBatchKruells9, cache, 1, times, x_callback)

    pls = PowerlawSeries(chain, var, cb, callback_kwargs={'param': default_params})
    pls.plot_datarow("figures/{}.pdf".format(this_name), formats.powerlaws, xlabel="${}$".format(var.human_name))
    #pls.plot_histograms("figures/{}_histograms.pdf".format(this_name), formats.doublehist)
    for v in var.values:
        nfig = NodeFigure(formats.doublehist)
        nfig.add(pls.chains[v]['histosetx'], 0, plot_on='spectra')
        nfig.add(pls.chains[v]['histosetp'], 1, plot_on='spectra')
        nfig.add(pls.chains[v]['powerlaw'], 1, plot_on='spectra')
        nfig.savefig("figures/{}_{}={}.pdf".format(name, var_name, v))

