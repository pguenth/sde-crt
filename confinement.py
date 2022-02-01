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

def cb(this_param):
    conf = this_param['xc']
    return {'confinements': [(0, lambda x : np.abs(x) < conf)], 'confinement_x' : conf}

def xparam_callback(common):
    kwargs = common['_kwargs_by_type']
    valueskw = [v for k, v in kwargs.items() if k is ValuesNode]
    return valueskw[0]['confinement_x']


name = 'confinement_9a'

param = { 
          't_inj' : 0.0035,
          'k_syn' : 0,#.0001,
          'x0' : 0,
          'y0' : 1,
        }

# judging from the dt study, 3000 timesteps should be enough
param_calc, param_num = chains.param_from_numerical(dx_adv=0.1, delta=0.5, sigma=0.5, beta_s=0.01, r=4, n_timesteps=5000)
param |= param_calc

cache = PickleNodeCache('pickle', name)
histo_opts = {'bin_count' : 30, 'plot' : 'hist', 'cache' : cache, 'ignore_cache' : False} 

batch = BatchNode('batch', batch_cls=PyBatchKruells9, cache=None, cache_not_found_action='ignore', param=param, ignore_cache=False)
points = PointNode('points', {'batch' : batch}, cache=cache, ignore_cache=False)

valuesx = ValuesNode('valuesx', {'points' : points}, index=0, cache=cache, ignore_cache=False)
histogramx = HistogramNode('histox', {'values' : valuesx}, log_bins=False, normalize='width', **histo_opts)

valuesp = ValuesNode('valuesp', index=1, cache=cache, ignore_cache=False)
histogramp = HistogramNode('histop', {'values' : valuesp}, log_bins=True, label='$X_c={confinement_x}$', normalize='density', **histo_opts)
powerlaw = PowerlawNode('pl', {'dataset' : histogramp}, plot='hist')
xparam_get = CommonCallbackNode('xparam_get', parents=histogramp, callback=xparam_callback)
datapoint_chain = NodeGroup('datapoint_chain', {'x' : xparam_get, 'y': powerlaw[1], 'dy' : powerlaw[3]})
dp_group = NodeGroup('group', {'datapoint' : datapoint_chain})

var = PowerlawSeriesVariable('x_\\textrm{conf}', 'xc', [0.01, 0.05, 0.1, 0.3, 0.5, 2, 5, 10])
var = PowerlawSeriesVariable('x_\\textrm{conf}', 'xc', [0.002, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 2, 5, 10, 50])

pls = PowerlawSeries(dp_group, var, cb, last_parents={'points' : points})
plgroup = NodeGroup('plgroup', [n.search_parent('pl') for n in pls.chains.values()])

nfmt = NodeFigureFormat(base=formats.histsandpowerlaw, fig_legend_kw={'loc': 'b', 'ncols': 6, 'order': 'F'}, axs_format={2: dict(xlabel='$x_\\textrm{conf}$', ylabel='Powerlaw index $s$')})
nfmt.legends_kw = {0: {}, 2: {}}
nfig = NodeFigure(nfmt)
nfig.format(suptitle='Influence of the confinement of particles on the powerlaw index')
nfig[2].format(xscale='log')
nfig.add(histogramx, 0, plot_on='hist')
nfig.add(plgroup, 1, plot_on='hist')
nfig.add(pls.datarow_chain, 2)
nfig.pad(0.2, 2)
nfig.savefig("figures/{}.pdf".format(name), fig_legend_kw={'handles' : plgroup.handles_complete_tree})
