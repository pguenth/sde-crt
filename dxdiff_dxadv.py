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
        dt = param['dt']
        delta = this_param['delta']
        dx_diff = this_param['dx_diff']

        dx_adv = delta * dx_diff
        beta_s = (dx_adv - dx_diff / 4) / dt
        q = dt / (dx_adv / dx_diff - 0.25)**2
        assert q > 0
        assert beta_s > 0
        Xsh = (dx_adv + dx_diff) / 2

        p = param | {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh}
        label_p = p | {'dx_adv' : dx_adv, 'dx_diff' : dx_diff}
        return {'param': p, 'label_fmt_fields': label_p}

    xlabel = '\\delta = x_\\textrm{adv}/x_\\textrm{diff}' if xaxis_delta else '\\Delta x_\\textrm{adv}'
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

    name = "dxdiff_dxadv_9a"
    x_diff = PowerlawSeriesVariable('\\Delta x_\\textrm{{diff}}', 'dx_diff', [0.5, 1, 5, 10])
    xvar = PowerlawSeriesVariable(xlabel, 'delta', [0.3, 0.4, 0.55, 0.7, 0.85, 0.99])

    cache = PickleNodeCache('pickle', name)
    chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells9, cache, 30, xparam_callback)
    pms = PowerlawMultiSeries(chain, x_diff, xvar, cb, name=name, callback_kwargs={'param': param})
    pms.plot_datarows("figures/{}.pdf".format(name), formats.powerlaws, xlabel='$' + xlabel + '$')
    pms.plot_histograms("figures", formats.doublehist)
    
fig_9a()
