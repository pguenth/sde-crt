import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.special.kruells import *
import logging

from powerlawstudy import *

pplt.rc.update({
        'text.usetex' : True,
        })

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def fig_9a(xaxis_delta=True):
    def cb(x_diff, delta, param, add_param):
        """
        Xdiff = 4sqrt(kappa dt)
        """
        dt = param['dt']
        x_adv = delta * x_diff
        beta_s = (x_adv - x_diff / 4) / dt
        q = dt / (x_adv / x_diff - 0.25)**2
        assert q > 0
        assert beta_s > 0
        Xsh = (x_adv + x_diff) / 2
        print(delta, x_adv, x_diff, beta_s, q, Xsh, sep=';\t')
        return {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh}, {'x_adv' : x_adv, 'x_diff' : x_diff}

    xlabel = '$\\delta = x_\\textrm{adv}/x_\\textrm{diff}$' if xaxis_delta else '$\\Delta x_\\textrm{adv}$'
    if xaxis_delta:
        xparam_callback = lambda c: c['label_fmt_fields']['x_adv'] / c['label_fmt_fields']['x_diff'] 
    else:
        xparam_callback = lambda c: c['label_fmt_fields']['x_adv']

    config = PowerlawStudyConfig(
        datarows_name = "xdiff",
        datarow_values = [0.5, 1, 5, 10],
        datarow_label = '$\\Delta x_\\textrm{{diff}}={}$',

        datapoints_name = "xadv",
        datapoint_values = [0.3, 0.4, 0.55, 0.7, 0.85, 0.99],
        datapoint_label = '$X_\\textrm{{adv}}={x_adv:.2f}$',

        xlabel = xlabel,
        xscale = None if xaxis_delta else 'log',

        datapoint_id_fmt_str = "xadv={x_adv}_xdiff={x_diff}",

        def_param =  { 
              'r' : 4,
              't_inj' : 0.001,
              'x0' : 0,
              'y0' : 1,
              'k_syn' : 0,
              'Tmax' : 300,
              'dt':  0.05
            },
        add_param = {},
        confine_x = 30,

        param_callback = cb,
        xparam_callback = xparam_callback 
    )

    pls = PowerlawStudy('xdiff_xadv_9a', PyBatchKruells9, config)
    pls.plot_datarows()
    pls.plot_momentum_spectra()


fig_9a()
