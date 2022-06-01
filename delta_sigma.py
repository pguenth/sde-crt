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


powerlawsnolegend = NodeFigureFormat(
                subplots=None,
                fig_format={'ylabel':'Power-law index $s$'},
                axs_format=[
                        {'xlabel': ''},
                ]
        )

def cb_9a(this_param, param, dx_adv):
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
    label_p = p | {'sigma' : sigma, 'invdelta': 1/delta, 'dx_adv' : dx_adv, 'dx_diff' : dx_diff}
    return {'param': p, 'label_fmt_fields': label_p}


def fig_9a():
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
    pms = PowerlawMultiSeries(chain, delta, sigma, cb_9a, cache=cache, name=name, callback_kwargs={'param': param, 'dx_adv' : 0.1})
    pms.plot_datarows("figures/{}.pdf".format(name), formats.powerlaws, xlabel='$\\sigma$', pad=0.2)
    pms.plot_histograms("figures", formats.doublehist)

def fig_9a_inv():
    param =  { 
          'r' : 4,
          't_inj' : 0.0005,
          'x0' : 0,
          'y0' : 1,
          'k_syn' : 0,
          'Tmax' : 300,
          'dt':  0.05
        }

    name = "delta_xsh_inverse_9a"

    #delta = PowerlawSeriesVariable('\\delta', 'delta', [float(sys.argv[1])])
    #sigma = PowerlawSeriesVariable('\\sigma', 'sigma', [float(sys.argv[2])])
    delta = PowerlawSeriesVariable('\\delta', 'delta', [0.3, 0.323, 0.35, 0.39, 0.4375, 0.5, 0.58, 0.7, 0.9])
    sigma = PowerlawSeriesVariable('\\sigma', 'sigma', [0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.95])

    cache = PickleNodeCache('pickle', name)
    chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells9, cache, 2, lambda c: c['label_fmt_fields']['invdelta'])
    pms = PowerlawMultiSeries(chain, sigma, delta, cb_9a, cache=cache, name=name, callback_kwargs={'param': param, 'dx_adv' : 0.1})
    pms.plot_histograms("figures", formats.doublehist)
    nfig = NodeFigure(powerlawsnolegend, xlabel='$\\delta^{-1}$', suptitle='Choice of shock width')
    nfig.add(pms.datarows_chain)
    for scnode in pms.datarows_chain.parents:
        x = max(scnode.data[0])
        y = scnode.data[1][np.argmax(scnode.data[0])]
        nfig[0].annotate(scnode.ext['label'], (x + 0.08, y - 0.012))

    nfig[0].invert_xaxis()
    nfig.pad(0.2, which='x')
    nfig.pad(0.4, which='r')
    nfig.savefig("figures/{}.pdf".format(name))


#def cb_14(this_param, param, dx_adv):
#    """
#    Xdiff = 4sqrt(kappa dt)
#    """
#    dt = param['dt']
#    invdelta = this_param['invdelta']
#    sigma = this_param['sigma']
#    dx_diff = dx_adv * invdelta
#    beta_s = (dx_adv - dx_diff / 4) / dt
#    q = dt / (dx_adv / dx_diff - 0.25)**2
#    assert q > 0
#    assert beta_s > 0
#    Xsh = dx_adv * (1 - sigma) + sigma * dx_diff
#    #printvars(dt=dt, x_adv=dx_adv, x_diff=dx_diff, beta_s=beta_s, q=q, Xsh=Xsh, delta=delta, sigma=sigma)
#    
#    p = param | {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh}
#    label_p = p | {'sigma' : sigma, 'invdelta': invdelta, 'dx_adv' : dx_adv, 'dx_diff' : dx_diff}
#    return {'param': p, 'label_fmt_fields': label_p}

def fig_14_inv():
    param =  { 
          'r' : 4,
          't_inj' : 0.05, #0.0005 = 7.5h (for euler scheme); 0.005 = 11.6h for semiimplicit
          'x0' : 0,
          'y0' : 1,
          'k_syn' : 0,
          'Tmax' : 300,
          'dt':  0.05
        }

    name = "delta_xsh_inverse_14"

    #delta = PowerlawSeriesVariable('\\delta', 'delta', [float(sys.argv[1])])
    #sigma = PowerlawSeriesVariable('\\sigma', 'sigma', [float(sys.argv[2])])
    delta = PowerlawSeriesVariable('\\delta', 'delta', [0.3, 0.4, 0.5, 0.6, 0.75, 0.9])
    sigma = PowerlawSeriesVariable('\\sigma', 'sigma', [0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.95])

    cache = PickleNodeCache('pickle', name)
    chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells14, cache, np.inf, lambda c: c['label_fmt_fields']['invdelta'], additional_confine_ranges=[(1, 1, np.inf)])
    pms = PowerlawMultiSeries(chain, sigma, delta, cb_9a, cache=cache, name=name, callback_kwargs={'param': param, 'dx_adv' : 5e-5})
    pms.plot_histograms("figures", formats.doublehist)
    nfig = NodeFigure(powerlawsnolegend, xlabel='$\\delta^{-1}$', title='Choice of shock width')
    nfig.add(pms.datarows_chain)
    for scnode in pms.datarows_chain.parents:
        x = max(scnode.data[0])
        y = scnode.data[1][np.argmax(scnode.data[0])]
        nfig[0].annotate(scnode.ext['label'], (x + 0.08, y - 0.012))

    nfig[0].invert_xaxis()
    nfig.pad(0.2, which='x')
    nfig.pad(0.4, which='r')
    nfig.savefig("figures/{}.pdf".format(name))

def fig_14_inv_2():
    # dxadv = 0.00053
    param =  { 
          'r' : 4,
          't_inj' : 0.1, #0.0005 = 7.5h (for euler scheme); 0.005 = 11.6h for semiimplicit
          'x0' : 0,
          'y0' : 1,
          'k_syn' : 0,
          'Tmax' : 300,
          'dt':  0.05
        }

    name = "delta_xsh_inverse_14_dxadv=0.00053"

    #delta = PowerlawSeriesVariable('\\delta', 'delta', [float(sys.argv[1])])
    #sigma = PowerlawSeriesVariable('\\sigma', 'sigma', [float(sys.argv[2])])
    delta = PowerlawSeriesVariable('\\delta', 'delta', [0.3, 0.5, 0.8])# [0.3, 0.4, 0.5, 0.6, 0.75, 0.9])
    sigma = PowerlawSeriesVariable('\\sigma', 'sigma', [0.95, 1.2, 2.0]) #[0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.95])

    cache = PickleNodeCache('pickle', name)
    chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells14, cache, np.inf, lambda c: c['label_fmt_fields']['invdelta'], additional_confine_ranges=[(1, 1, np.inf)])
    pms = PowerlawMultiSeries(chain, sigma, delta, cb_9a, cache=cache, name=name, callback_kwargs={'param': param, 'dx_adv' : 0.00053})
    pms.plot_histograms("figures", formats.doublehist)
    nfig = NodeFigure(powerlawsnolegend, xlabel='$\\delta^{-1}$', title='Choice of shock width')
    nfig.add(pms.datarows_chain)
    for scnode in pms.datarows_chain.parents:
        x = max(scnode.data[0])
        y = scnode.data[1][np.argmax(scnode.data[0])]
        nfig[0].annotate(scnode.ext['label'], (x + 0.08, y - 0.012))

    nfig[0].invert_xaxis()
    nfig.pad(0.2, which='x')
    nfig.pad(0.4, which='r')
    nfig.savefig("figures/{}.pdf".format(name))
   
#fig_9a()
fig_9a_inv()
#fig_14_inv()
#fig_14_inv_2()
