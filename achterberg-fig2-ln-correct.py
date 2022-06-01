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

def cb(this_param, param, other_param):
    dt = other_param['alpha'] / param['V']
    q = param['V'] * other_param['Ldiff'] 
    Ls = other_param['Ldiff'] * this_param['epsilon']
    batch_param = param | {'q' : q, 'dt' : dt, 'Ls' : Ls}
    label_fmt_fields = this_param | other_param
    return {'param': batch_param, 'label_fmt_fields': label_fmt_fields}

param = {
          'r' : 4,
          'x0' : 0,
          'y0' : 0,
          't_inj' : 1, #0.1, # 0.2
          'Tmax' : 3500,
          'V' : 1,
        }
other_param = {
          'alpha' : 0.05,
          'Ldiff' : 1,
        }

name = 'achterberg_fig2_ln_correct'
cache = PickleNodeCache('pickle', name)
add_conf = [(1, 0, np.inf)]
chain = chains.get_chain_powerlaw_datapoint(PyBatchAchterberg2, cache, np.inf, lambda c: c['label_fmt_fields']['epsilon'], histo_opts={'bin_count' : 40}, negate_index=True, log_bins=False, additional_confine_ranges=add_conf)
#chain.map_tree(lambda n: setattr(n, 'ignore_cache', True), "values")
chain.map_tree(lambda n: n.set(ln_x=True) , "pl")
chain.map_tree(lambda n: print(n.plot_on), "pl")

#var = PowerlawSeriesVariable('\\epsilon', 'epsilon', [0.01, 0.02, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1])
#var = PowerlawSeriesVariable('\\epsilon', 'epsilon', [0.01, 0.04, 0.08, 0.2, 0.4, 0.8, 1.0]) #[0.01, 0.05, 0.1, 0.8])[float(sys.argv[1])])
var = PowerlawSeriesVariable('\\epsilon', 'epsilon', [float(sys.argv[1])])
#var = PowerlawSeriesVariable('\\epsilon', 'epsilon', [0.01 * int(sys.argv[1])])
pls = PowerlawSeries(chain, var, cb, 
            callback_kwargs={'param': param, 'other_param': other_param}
        )

axs_format = SliceDict()
axs_format[2] = dict(xlabel='ln p/p_inj', xscale='linear')
axs_format[3] = dict(xlabel='ln p/p_inj', xscale='linear')
axs_format[4] = dict(xlabel='$\\epsilon$', ylabel='Powerlaw index $s$', xscale='log')
axs_format[slice(None, None, None)] = dict(toplabels=('Spatial distribution', 'Momentum distribution', 'Powerlaw indizes'), leftlabels=('CES', 'KPPC'))
nfmt = NodeFigureFormat(base=formats.histsandpowerlaw2, fig_legend_kw=None, axs_format=axs_format, legends_kw={0: None, 1: None, 2: None, 4:{}})
nfig = NodeFigure(nfmt)
nfig.format(suptitle='Fig. 2 of Achterberg/Schure')
chain_x, chain_p = pls.histogram_chains

datarow = pls.datarow_chain
datarow.set(label='Cauchy-Euler scheme')


def mod_scheme(pls, batch_cls, name, name_human):
        memo = {}
        chain_x_old, chain_p_old = pls.histogram_chains
        datarow_old = pls.datarow_chain
        chain_x = chain_x_old.copy(name, last_kwargs={'batch_cls': batch_cls}, memo=memo)
        chain_p = chain_p_old.copy(name, last_kwargs={'batch_cls': batch_cls}, memo=memo)
        datarow = datarow_old.copy(name, last_parents={'batch_cls': batch_cls}, memo=memo)
        datarow.set(label=name_human)
        return datarow, chain_x, chain_p

datarow_semiimplicit, chain_x_semiimplicit, chain_p_semiimplicit = mod_scheme(pls, PyBatchAchterberg2SemiImplicit2, 'semiimplicit', 'Semi-implicit')
datarow_implicit, chain_x_implicit, chain_p_implicit = mod_scheme(pls, PyBatchAchterberg2Implicit, 'implicit', 'Implicit')
datarow_kppc, chain_x_kppc, chain_p_kppc = mod_scheme(pls, PyBatchAchterberg2KPPC, 'kppc', 'KPPC')

#memo = {}
#secondorder_cls = PyBatchAchterberg2SecondOrder
#chain_x_2nd = chain_x.copy("2ndorder", last_kwargs={'batch_cls': secondorder_cls}, memo=memo)
#chain_p_2nd = chain_p.copy("2ndorder", last_kwargs={'batch_cls': secondorder_cls}, memo=memo)
#datarow_2nd = datarow.copy("2ndorder", last_parents={'batch_cls': secondorder_cls}, memo=memo)
#datarow_2nd.set(label='2nd order scheme')
#
#memo = {}
#secondorder2_cls = PyBatchAchterberg2SecondOrder2
#chain_x_2nd2 = chain_x.copy("2ndorder2", last_kwargs={'batch_cls': secondorder2_cls}, memo=memo)
#chain_p_2nd2 = chain_p.copy("2ndorder2", last_kwargs={'batch_cls': secondorder2_cls}, memo=memo)
#datarow_2nd2 = datarow.copy("2ndorder2", last_parents={'batch_cls': secondorder2_cls}, memo=memo)
#datarow_2nd2.set(label='2nd order scheme (vec)')

#memo = {}
#semiimplicit_cls = PyBatchAchterberg2SemiImplicit
#chain_x_semiimplicit = chain_x.copy("semiimplicit", last_kwargs={'batch_cls': semiimplicit_cls}, memo=memo)
#chain_p_semiimplicit = chain_p.copy("semiimplicit", last_kwargs={'batch_cls': semiimplicit_cls}, memo=memo)
#datarow_semiimplicit = datarow.copy("semiimplicit", last_parents={'batch_cls': semiimplicit_cls}, memo=memo)
#datarow_semiimplicit.set(label='semiimplicit scheme (vec)')

#nfig.add(chain_x, 0, plot_on='spectra')
#nfig.add(chain_p, 2, plot_on='spectra')
nfig.add(datarow, 4)

#nfig.add(chain_x_kppc, 1, plot_on='spectra')
#nfig.add(chain_p_kppc, 3, plot_on='spectra')
nfig.add(datarow_kppc, 4)
#
nfig.add(chain_x_implicit, 0, plot_on='spectra')
nfig.add(chain_p_implicit, 2, plot_on='spectra')
nfig.add(datarow_implicit, 4)

comparison_function = lambda eps : 1 + 0.924 * eps + 0.095 * eps**2
comparison_node = FunctionNode("fun", callback=comparison_function, plot=True)
nfig.add(comparison_node, 4)

#nfig.add(chain_x_2nd2, 1, plot_on='spectra')
#nfig.add(chain_p_2nd2, 3, plot_on='spectra')
#nfig.add(datarow_2nd2, 4)

#nfig.add(chain_x_2nd, 1, plot_on='spectra')
#nfig.add(chain_p_2nd, 3, plot_on='spectra')
#nfig.add(datarow_2nd, 4)

nfig.add(chain_x_semiimplicit, 0, plot_on='spectra')
nfig.add(chain_p_semiimplicit, 2, plot_on='spectra')
nfig.add(datarow_semiimplicit, 4)


nfig.pad(0.2, 4)
nfig[2].legend()
nfig[3].legend()
#nfig.show_nodes("achterberg_tree.pdf")
nfig.savefig("figures/{}.pdf".format(name))

#nfig = NodeFigure(NodeFigureFormat(subplots={'ncols' : 1}, ax_format={'xscale': 'log', 'xlabel': 'streaming velocity / diffusivity', 'ylabel': 'powerlaw index'}, legend_kw={'ncols': 1}))
#nfig.add(pls.datarow_chain)
#nfig.add(datarow_kppc)
#nfig.add(datarow_implicit)
#nfig.add(comparison_node)
#nfig.savefig("figures/{}-simple.pdf".format(name))
