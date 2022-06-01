import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.special.kruells import *
import proplot as pplt
import logging
import chains
import formats
from node.cache import PickleNodeCache
from matplotlib import ticker

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
# dxadv(') = 0.09975
# dxdiff = 0.285
# delta = 0.35
# sigma = 0.811067


name = 'boundary_11'
cache = PickleNodeCache('pickle', name)
chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells11, cache, 2, lambda c: c['batch_param']['L'])

var = PowerlawSeriesVariable('L', 'L', [3, 5, 6, 8, 10, 20, 30, 50, 80])#, 200, 1000])
pls = PowerlawSeries(chain, var, cb, callback_kwargs={'param': param})
pls.plot_datarow("figures/{}.pdf".format(name), formats.powerlaws, xlabel='$L$', xscale='log')
nfig = NodeFigure(formats.powerlaws, title='Spectral steepening at free-escape boundary', xlabel='Boundary radius (from shock) $X_C$', xscale='log')
nfig.add(pls.datarow_chain)
nfig.pad(.2)

# endlevel hack for minor ticks
logf = ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(10, 0.4))
nfig[0].xaxis.set_minor_formatter(logf)
def formatterfunc(x, pos):
        logfn = logf(x, pos)
        if logfn[0] in ['5', '7', '8', '9']:
                return ""
        else:
                return "$" + logf(x, pos) + "$"
nfig[0].xaxis.set_minor_formatter(ticker.FuncFormatter(formatterfunc))
nfig[0].xaxis.set_minor_locator(ticker.LogLocator(subs='all'))
# ----------------------------

nfig[0].annotate('$\\Delta x_\\mathrm{diff} = 0.285,~~\\delta =0.35,~~\\sigma\\approx0.81$', (0.32, 0.18), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=1))
nfig._legends_kw = {}
nfig.savefig("figures/{}.pdf".format(name))
pls.plot_histograms("figures/{}_histograms.pdf".format(name), formats.doublehist)
