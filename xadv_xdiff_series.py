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
    def cb(x_adv, invdelta, param, add_param):
        """
        Xdiff = 4sqrt(kappa dt)
        """
        dt = param['dt']
        x_diff = invdelta * x_adv
        beta_s = (x_adv - x_diff / 4) / dt
        q = dt / (x_adv / x_diff - 0.25)**2
        assert q > 0
        assert beta_s > 0
        Xsh = (x_adv + x_diff) / 2
        
        return {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh}, {'x_adv' : x_adv, 'x_diff' : x_diff}

    xlabel = '$\\delta^{-1} = x_\\textrm{diff}/x_\\textrm{adv}$' if xaxis_delta else '$\\Delta x_\\textrm{diff}$'
    if xaxis_delta:
        xparam_callback = lambda c: c['label_fmt_fields']['x_diff'] / c['label_fmt_fields']['x_adv'] 
    else:
        xparam_callback = lambda c: c['label_fmt_fields']['x_diff']

    name = "xdiff_xadv_9a"
    x_advs = [0.0005, 0.005, 0.05, 0.1, 0.5, 1],
    x_diffs = [1.01, 1.1, 1.2, 1.5, 1.8, 2.4, 3.0, 3.2, 3.4, 3.6],
    def_param =  { 
          'r' : 4,
          't_inj' : 0.001,
          'x0' : 0,
          'y0' : 1,
          'k_syn' : 0,
          'Tmax' : 300,
          'dt':  0.05
        },
    series = {}
    for x_adv in x_advs:
        var = PowerlawSeriesVariable("X_\\textrm{{diff}}", "xdiff", x_diffs)
        series[x_adv] = PowerlawSeries("xadv={}".format(x_adv), PyBatchKruells9, var, 


    config_9a = PowerlawStudyConfig(
        datarows_name = "xadv",
        datarow_label = '$\\Delta x_\\textrm{{adv}}={}$',

        datapoints_name = "xdiff",
        datapoint_label = '$X_\\textrm{{diff}}={x_diff:.2f}$',

        xlabel = xlabel,
        xscale = None if xaxis_delta else 'log',

        datapoint_id_fmt_str = "xadv={x_adv}_xdiff={x_diff}",

        add_param = {},
        confine_x = 10, 

        param_callback = cb,
        xparam_callback = xparam_callback
    )

    pls = PowerlawStudy('xadv_xdiff_9a', PyBatchKruells9, config_9a)
    pls.plot_datarows()
    pls.plot_momentum_spectra()

fig_9a()
