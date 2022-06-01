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
        """
        Xdiff = 4sqrt(kappa dt)
        """
        dx_adv = param['delta'] * this_param['dx_diff']
        param_calc, param_num = chains.param_from_numerical(dx_adv=dx_adv, delta=param['delta'], sigma=param['sigma'], beta_s=param['beta_s'], r=param['r'], n_timesteps=param['n_timesteps'])
        t_inj = param_calc['Tmax'] / param['nparticles']

        p = param | param_calc | {'t_inj' : t_inj, 'L' : this_param['L']}
        label_p = p | {'dx_adv' : dx_adv, 'dx_diff' : param_num['dx_diff']}
        return {'param': p, 'label_fmt_fields': label_p}

param = { 
      #'t_inj' : 0.0035,
      'nparticles' : 50000,
      'k_syn' : 0,#.0001,
      'x0' : 0,
      'y0' : 1,
    }

# invdelta = 3.1
param |= dict(delta=0.323, sigma=0.8, beta_s=1, r=4, n_timesteps=5000)

name = 'boundary_multi_11'
cache = PickleNodeCache('pickle', name)
chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells11, cache, 2, lambda c: c['batch_param']['L'])

xdiff = PowerlawSeriesVariable('', 'dx_diff', [0.01, 0.05, 0.5, 2])
varL = PowerlawSeriesVariable('L', 'L', [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 6, 8, 10, 20, 30, 50, 80, 100, 200, 500, 1000, 2000])#, 200, 1000])

pms = PowerlawMultiSeries(chain, xdiff, varL, cb, name=name, callback_kwargs={'param': param})
pms.datarows_chain.map_tree(lambda s: s.set(value_limits=(-np.inf, -2)), 'scatter')

cutoffs = {}
for sc in pms.datarows_chain.search_tree_all("scatter"):
        coname = sc.name[8:]
        cutoffs[coname] = CutoffFindNode("cutoff_" + coname, parents=sc, plot=True, reverse=True, sigmas=12)

cutoffg = NodeGroup("cutoffg", parents=cutoffs)
        
nfig = NodeFigure(formats.powerlaws, suptitle='Steeper spectra at free-escape boundary', xlabel='Boundary radius (from shock) $X_C$', xscale='log')
nfig.add(pms.datarows_chain)
#nfig.add(cutoffg)
nfig.pad(.2)
nfig[0].format(ylim=(-8, -1.8), xlim=(0.07, 2200),xformatter=ticker.LogFormatter())

# endlevel hack for minor ticks
#logf = ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(10, 0.4))
#nfig[0].xaxis.set_minor_formatter(logf)
#def formatterfunc(x, pos):
#        logfn = logf(x, pos)
#        if logfn[0] in ['5', '7', '8', '9']:
#                return ""
#        else:
#                return "$" + logf(x, pos) + "$"
#nfig[0].xaxis.set_minor_formatter(ticker.FuncFormatter(formatterfunc))
#nfig[0].xaxis.set_minor_locator(ticker.LogLocator(subs='all'))
# ----------------------------

nfig[0].annotate('$\\delta =0.323,~~\\sigma=0.8$', (0.2, 0.87), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
nfig._legends_kw = {}
nfig[0].legend(title='$\\Delta x_\\mathrm{diff}$', ncols=1)
nfig.savefig("figures/{}.pdf".format(name))
#pms.plot_histograms("figures", formats.doublehist)
