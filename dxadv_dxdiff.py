import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.special.kruells import *
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

def fig_9a(xaxis_delta=True):
    def cb(this_param, param):
        """
        Xdiff = 4sqrt(kappa dt)
        """
        dt = param['dt']
        dx_adv = this_param['dx_adv']
        invdelta = this_param['invdelta']
        dx_diff = invdelta * dx_adv

        beta_s = (dx_adv - dx_diff / 4) / dt
        q = dt / (dx_adv / dx_diff - 0.25)**2
        assert q > 0
        assert beta_s > 0
        Xsh = (dx_adv + dx_diff) / 2
        
        p = param | {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh}
        label_p = p | {'dx_adv' : dx_adv, 'dx_diff' : dx_diff}
        return {'param': p, 'label_fmt_fields': label_p}

    xlabel = '\\delta^{-1} = \\Delta x_\\textrm{diff}/\\Delta x_\\textrm{adv}\'' if xaxis_delta else '\\Delta x_\\textrm{diff}'
    if xaxis_delta:
        xparam_callback = lambda c: c['label_fmt_fields']['dx_diff'] / c['label_fmt_fields']['dx_adv'] 
    else:
        xparam_callback = lambda c: c['label_fmt_fields']['dx_diff']

    param =  { 
          'r' : 4,
          't_inj' : 0.001,
          'x0' : 0,
          'y0' : 1,
          'k_syn' : 0,
          'Tmax' : 300,
          'dt':  0.05
        }

    name = "dxadv_dxdiff_9a"
    x_adv = PowerlawSeriesVariable('\\Delta x_\\textrm{{adv}}\'', 'dx_adv', [0.0005, 0.005, 0.05, 0.1, 0.5, 1])
    xvar = PowerlawSeriesVariable(xlabel, 'invdelta', [1.01, 1.1, 1.2, 1.5, 1.8, 2.4, 3.0, 3.2, 3.4, 3.6])

    cache = PickleNodeCache('pickle', name)
    chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells9, cache, 10, xparam_callback)
    pms = PowerlawMultiSeries(chain, x_adv, xvar, cb, name=name, callback_kwargs={'param': param})
    nfig = NodeFigure(formats.powerlaws, title='Influence of the ratio of step sizes on the power law index', xlabel='$' + xlabel + '$')
    nfig.add(pms.datarows_chain)
    nfig.pad(.2)
    nfig.savefig("figures/{}.pdf".format(name), legend_kw={'ncols': 1})
    pms.plot_histograms("figures", formats.doublehist)

def fig_12(xaxis_delta=True):
    def cb(this_param, param):
        """
        Xdiff = 4sqrt(kappa dt)
        """
        dt = param['dt']
        dx_adv = this_param['dx_adv']
        invdelta = this_param['invdelta']
        dx_diff = invdelta * dx_adv

        beta_s = (dx_adv - dx_diff / 4) / dt
        q = dt / (dx_adv / dx_diff - 0.25)**2
        assert q > 0
        assert beta_s > 0
        Xsh = (dx_adv + dx_diff) / 2
        
        p = param | {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh, 'Xdiff' : 4 * dx_diff}
        label_p = p | {'dx_adv' : dx_adv, 'dx_diff' : dx_diff}
        return {'param': p, 'label_fmt_fields': label_p}

    xlabel = '\\delta^{-1} = \\Delta x_\\textrm{diff}/\\Delta x_\\textrm{adv}\'' if xaxis_delta else '\\Delta x_\\textrm{diff}'
    if xaxis_delta:
        xparam_callback = lambda c: c['label_fmt_fields']['dx_diff'] / c['label_fmt_fields']['dx_adv'] 
    else:
        xparam_callback = lambda c: c['label_fmt_fields']['dx_diff']

    param =  { 
          'r' : 4,
          't_inj' : 0.001,
          'x0' : 0,
          'y0' : 1,
          'k_syn' : 0,
          'Tmax' : 150,
          'dt':  0.05,
        }

    name = "dxadv_dxdiff_12"
    x_adv = PowerlawSeriesVariable('\\Delta x_\\textrm{{adv}}\'', 'dx_adv', [0.0005, 0.05, 0.5, 5])
    xvar = PowerlawSeriesVariable(xlabel, 'invdelta', [1.01, 1.5, 2.0, 3.0, 3.5])

    cache = PickleNodeCache('pickle', name)
    chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells12, cache, 10, xparam_callback)
    pms = PowerlawMultiSeries(chain, x_adv, xvar, cb, name=name, callback_kwargs={'param': param})
    nfig = NodeFigure(formats.powerlaws, title='Influence of the ratio of step sizes on the power law index with eq. 19', xlabel='$' + xlabel + '$')
    nfig.add(pms.datarows_chain)
    nfig.pad(.2)
    nfig.savefig("figures/{}.pdf".format(name), legend_kw={'ncols': 1})
    pms.plot_histograms("figures", formats.doublehist)
    

fig_12()
