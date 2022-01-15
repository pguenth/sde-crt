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

def printvars(**kwargs):
    s = ""
    for k, v in kwargs.items():
        s += "{}={}, ".format(k, v)

    print(s)

def fig_9a():
    def cb(delta, sigma, param, add_param):
        """
        Xdiff = 4sqrt(kappa dt)
        """
        dt = param['dt']
        x_adv = add_param['x_adv']
        x_diff = x_adv / delta
        beta_s = (x_adv - x_diff / 4) / dt
        q = dt / (x_adv / x_diff - 0.25)**2
        assert q > 0
        assert beta_s > 0
        Xsh = x_adv * (1 - sigma) + sigma * x_diff
        printvars(dt=dt, x_adv=x_adv, x_diff=x_diff, beta_s=beta_s, q=q, Xsh=Xsh, delta=delta, sigma=sigma)
        
        return {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh}, {'sigma' : sigma, 'x_adv' : x_adv, 'x_diff' : x_diff}

    config = PowerlawStudyConfig(
        datarows_name = "delta",
        datarow_values = [0.35, 0.5, 0.7, 0.9],
        datarow_label = '$\\delta={}$',#'$\\Delta x_\\textrm{{diff}}={}$',

        datapoints_name = "sigma",
        datapoint_values = [0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.95],
        datapoint_label = '$\\sigma={sigma:.2f}$',

        xlabel = '$\\sigma$',
        xscale = None, 

        datapoint_id_fmt_str = "x_diff={x_diff}_Xsh={Xsh}",

        def_param =  { 
              'r' : 4,
              't_inj' : 0.0005,
              'x0' : 0,
              'y0' : 1,
              'k_syn' : 0,
              'Tmax' : 300,
              'dt':  0.05
            },
        
        add_param = {
            'x_adv' : 0.1,
        },
        confine_x = 2,

        param_callback = cb,
        xparam_callback = lambda c: c['label_fmt_fields']['sigma']
    )

    pls = PowerlawStudy('xdiff_xsh_9a', PyBatchKruells9, config)
    pls.plot_datarows()
    pls.plot_momentum_spectra()

fig_9a()
