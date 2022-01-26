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

def printvars(**kwargs):
    s = ""
    for k, v in kwargs.items():
        s += "{}={}, ".format(k, v)

    print(s)

def fig_9a():
    def cb(this_param, param, dx_adv):
        """
        Xdiff = 4sqrt(kappa dt)
        """
        dt = param['dt']
        delta = this_param['delta']
        sigma = this_param['sigma']
        dx_diff = dx_adv / delta
        beta_s = (dx_adv - dx_diff / 4) / dt
        q = dt / (dx_adv / dx_diff - 0.25)**2
        assert q > 0
        assert beta_s > 0
        Xsh = dx_adv * (1 - sigma) + sigma * dx_diff
        #printvars(dt=dt, x_adv=dx_adv, x_diff=dx_diff, beta_s=beta_s, q=q, Xsh=Xsh, delta=delta, sigma=sigma)
        
        p = param | {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh}
        label_p = p | {'sigma' : sigma, 'dx_adv' : dx_adv, 'dx_diff' : dx_diff}
        return {'param': p, 'label_fmt_fields': label_p}

    param =  { 
          'r' : 4,
          't_inj' : 0.0005,
          'x0' : 0,
          'y0' : 1,
          'k_syn' : 0,
          'Tmax' : 300,
          'dt':  0.05
        }

    name = "delta_xsh_9a"
    delta = PowerlawSeriesVariable('\\delta', 'delta', [0.35, 0.5, 0.7, 0.9])
    sigma = PowerlawSeriesVariable('\\sigma', 'sigma', [0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.95])

    cache = PickleNodeCache('pickle', name)
    chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells9, cache, 2, lambda c: c['label_fmt_fields']['sigma'])
    pms = PowerlawMultiSeries(chain, delta, sigma, cb, name=name, callback_kwargs={'param': param, 'dx_adv' : 0.1})
    pms.plot_datarows("figures/{}.pdf".format(name), formats.powerlaws, xlabel='$\\sigma$')
    pms.plot_histograms("figures", formats.doublehist)

fig_9a()
