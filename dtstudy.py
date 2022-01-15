import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.special.kruells import *
import proplot as pplt
import logging

from powerlawstudy import PowerlawSeries

pplt.rc.update({
        'text.usetex' : True,
        })

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def dtstudy_cb(this_param, def_param, abstract_param):
    dt = this_param['dt']
    x_adv = abstract_param['x_adv_per_dt'] * dt
    delta = abstract_param['delta']
    sigma = abstract_param['sigma']
    x_diff = x_adv / delta
    beta_s = (x_adv - x_diff / 4) / dt
    q = dt / (x_adv / x_diff - 0.25)**2
    Xsh = x_adv * (1 - sigma) + sigma * x_diff

    assert q > 0
    assert beta_s > 0
    assert Xsh > 0

    return {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh, 'dt' : dt}, {'delta' : delta, 'sigma' : sigma, 'x_adv' : x_adv, 'x_diff' : x_diff}

pls = PowerlawSeries('dtstudy_9a', PyBatchKruells9, 'dt', [0.00005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 2, 5, 10],
            {
              'r' : 4,
              't_inj' : 0.005,
              'x0' : 0,
              'y0' : 1,
              'k_syn' : 0,
              'Tmax' : 300,
            },
            {
                'delta' : 0.5,
                'sigma' : 0.5,
                'x_adv_per_dt' : 0.5
            },
            param_callback = dtstudy_cb,
            confine_x = 5,
            reeval = False,
            param_human_name = "\\Delta\\tau"
        )
pls.get_series()
pls.get_histograms()
