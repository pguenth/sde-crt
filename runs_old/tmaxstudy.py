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

def tmaxstudy_cb(this_param, param, other_param):
    dt = param['dt']
    x_adv = other_param['x_adv'] 
    delta = other_param['delta']
    sigma = other_param['sigma']
    t_inj = this_param['Tmax'] / other_param['Nparticle']
    x_diff = x_adv / delta
    beta_s = (x_adv - x_diff / 4) / dt
    q = dt / (x_adv / x_diff - 0.25)**2
    Xsh = x_adv * (1 - sigma) + sigma * x_diff

    assert q > 0
    assert beta_s > 0
    assert Xsh > 0

    batch_param = param | {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh, 'dt' : dt, 't_inj' : t_inj, 'Tmax' : this_param['Tmax']}
    label_fmt_fields = {'delta' : delta, 'sigma' : sigma, 'x_adv' : x_adv, 'x_diff' : x_diff}
    return {'param': batch_param, 'label_fmt_fields': label_fmt_fields}

def get_tmaxstudy(name, batch_cls):
        param = {
                  'r' : 4,
                  'x0' : 0,
                  'y0' : 1,
                  'k_syn' : 0,
                  'dt' : 0.1,
                }
        other_param = {
                    'delta' : 0.5,
                    'sigma' : 0.5,
                    'x_adv' : 0.1,
                    'Nparticle' : 10000
                }


        name = 'tmaxstudy_' + name
        cache = PickleNodeCache('pickle', name)
        chain = chains.get_chain_powerlaw_datapoint(batch_cls, cache, np.inf, lambda c: c['batch_param']['Tmax'])

        var = PowerlawSeriesVariable('T', 'Tmax', [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
        pls = PowerlawSeries(chain, var, tmaxstudy_cb, 
                    callback_kwargs={'param': param, 'other_param': other_param}
                )

        return pls

if __name__ == "__main__":
        name = "9a"
        pls = get_tmaxstudy(name, PyBatchKruells9)
        """
        name = "14"
        pls = get_tmaxstudy(name, PyBatchKruells14)
        """

        nfig = NodeFigure(formats.powerlaws, title='Reaching the temporal equilibrium', xlabel='Runtime $T$', xscale='log', 
                        xformatter=pplt.SimpleFormatter(precision=3, prefix="$", suffix="$"))
        nfig.add(pls.datarow_chain)
        nfig.pad(.2)
        nfig[0].annotate('$\\delta =0.5,~~\\sigma=0.5,~~\\Delta x_\\mathrm{adv}=\\Delta\\tau/2$', (0.22, 0.18), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=1))
        nfig._legends_kw = {}
        nfig.savefig("figures/tmaxstudy_{}.pdf".format(name))
        pls.plot_histograms("figures/tmaxstudy_{}_histograms.pdf".format(name), formats.doublehist)
