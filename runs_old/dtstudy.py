import sys
sys.path.insert(0, 'lib')
from pybatch.special.kruells import *
import proplot as pplt
import logging
import chains
import formats
from grapheval.cache import PickleNodeCache

from powerlawseries import *

pplt.rc.update({
        'text.usetex' : True,
        })

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def dtstudy_cb(this_param, param, other_param):
    dt = this_param['dt']
    x_adv = other_param['x_adv_per_dt'] * dt
    delta = other_param['delta']
    sigma = other_param['sigma']
    x_diff = x_adv / delta
    beta_s = (x_adv - x_diff / 4) / dt
    q = dt / (x_adv / x_diff - 0.25)**2
    Xsh = x_adv * (1 - sigma) + sigma * x_diff

    assert q > 0
    assert beta_s > 0
    assert Xsh > 0

    batch_param = param | {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh, 'dt' : dt}
    label_fmt_fields = {'delta' : delta, 'sigma' : sigma, 'x_adv' : x_adv, 'x_diff' : x_diff}
    return {'param': batch_param, 'label_fmt_fields': label_fmt_fields}

def get_dtstudy(name, batch_cls):
        param = {
                  'r' : 4,
                  't_inj' : 0.05, #0.005
                  'x0' : 0,
                  'y0' : 1,
                  'k_syn' : 0,
                  'Tmax' : 300,
                }
        other_param = {
                    'delta' : 0.5,
                    'sigma' : 0.5,
                    'x_adv_per_dt' : 0.5
                }

        name = 'dtstudy_' + name
        cache = PickleNodeCache('pickle', name)
        chain = chains.get_chain_powerlaw_datapoint(batch_cls, cache, 5, lambda c: c['batch_param']['dt'])

        var = PowerlawSeriesVariable('\\Delta\\tau', 'dt', [0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 2, 5, 10])#0.00005, 0.001, 0.005, 0.01
        pls = PowerlawSeries(chain, var, dtstudy_cb, 
                    callback_kwargs={'param': param, 'other_param': other_param}
                )

        return pls

if __name__ == "__main__":
        """
        name = "14"
        pls = get_dtstudy(name, PyBatchKruells14)
        """
        name = "9a"
        pls = get_dtstudy(name, PyBatchKruells9)
        nfig = NodeFigure(formats.powerlaws, suptitle='Reaching the temporal equilibrium', xlabel='Timestep $\\Delta\\tau$', xscale='log', 
                        xformatter=pplt.SimpleFormatter(precision=3, prefix="$", suffix="$"))
        nfig.add(pls.datarow_chain)
        nfig.pad(.2)
        nfig[0].annotate('$\\delta =0.5,~~\\sigma=0.5,~~\\Delta x_\\mathrm{adv}=\\Delta\\tau/2$', (0.22, 0.18), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=1))
        nfig._legends_kw = {}
        nfig.savefig("figures/dtstudy_{}.pdf".format(name))
        pls.plot_histograms("figures/dtstudy_{}_histograms.pdf".format(name), formats.doublehist)
