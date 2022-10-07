import sys
sys.path.insert(0, 'lib')
from pybatch.special.kruells import *
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

def fig_kruells_style_full(batch_cls, name, dx_advs, invdeltas, param, xaxis_delta=True, spectra_xlim=None, spectra_plim=None):
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

        if 'sigma' in param:
            s = param['sigma']
            Xsh = (1 - s) * dx_adv + s * dx_diff
        else:
            Xsh = (dx_adv + dx_diff) / 2
        
        p = param | {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh}
        label_p = p | {'dx_adv' : dx_adv, 'dx_diff' : dx_diff}
        return {'param': p, 'label_fmt_fields': label_p}

    xlabel = '\\delta^{-1} = \\Delta x_\\textrm{diff}/\\Delta x_\\textrm{adv}' if xaxis_delta else '\\Delta x_\\textrm{diff}'
    if xaxis_delta:
        xparam_callback = lambda c: c['label_fmt_fields']['dx_diff'] / c['label_fmt_fields']['dx_adv'] 
    else:
        xparam_callback = lambda c: c['label_fmt_fields']['dx_diff']

    name = "dxadv_dxdiff_" + name
    x_adv = PowerlawSeriesVariable('', 'dx_adv', dx_advs)
    xvar = PowerlawSeriesVariable(xlabel, 'invdelta', invdeltas)

    cache = PickleNodeCache('pickle', name)
    chain = chains.get_chain_powerlaw_datapoint(batch_cls, cache, np.inf, xparam_callback, additional_confine_ranges=[(1, 1, np.inf)])
    pms = PowerlawMultiSeries(chain, x_adv, xvar, cb, name=name, callback_kwargs={'param': param})
    nfig = NodeFigure(formats.powerlaws, title='Ratio of diffusive and advective step lengths', xlabel='$' + xlabel + '$')
    nfig.add(pms.datarows_chain)
    nfig.pad(.2)
    nfig[0].annotate('$\\Delta\\tau =' + str(param['dt']) + '$\n$\\sigma=0.5$', (0.50, 0.18), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
    nfig[0].legend(ncols=1, title='$\\Delta x_\\textrm{adv}$')
    nfig._legends_kw = {}
    nfig.savefig("figures/{}.pdf".format(name))

    for val, ps in pms.series_dict.items():
        chain_x, chain_p = ps.histogram_chains
        nfig = NodeFigure(formats.doublehist, suptitle="Histograms for $\\Delta x_\\textrm{{adv}}={}$".format(val))
        if not spectra_xlim is None:
            def mod_h(h):
                dxadv = float(h.name.split('=')[1].split('_')[0])
                bw = (spectra_xlim[dxadv][1] - spectra_xlim[dxadv][0]) / 40
                h.set(bin_width=bw)
            chain_x.map_tree(mod_h, starts_with="histox")
            nfig[0].format(xlim=spectra_xlim[val])

        if not spectra_plim is None:
            nfig[1].format(xlim=spectra_plim)

        nfig.add(chain_x, 0, plot_on='spectra')
        nfig.add(chain_p, 1, plot_on='spectra')
        nfig.savefig("figures/{}_histograms_dxadv={}.pdf".format(name, val))

def fig_kruells_style_full_inv(batch_cls, name, dx_advs, invdeltas, param, spectra_xlim=None, spectra_plim=None):
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

        if 'sigma' in param:
            s = param['sigma']
            Xsh = (1 - s) * dx_adv + s * dx_diff
        else:
            Xsh = (dx_adv + dx_diff) / 2
        
        p = param | {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh}
        label_p = p | {'dx_adv' : dx_adv, 'dx_diff' : dx_diff}
        return {'param': p, 'label_fmt_fields': label_p}

    xlabel = '\\delta^{{-1}} = \\Delta x_\\textrm{{diff}}/\\Delta x_\\textrm{{adv}}'
    xlabel = '\\Delta x_\\textrm{{adv}}'
    xparam_callback = lambda c: np.log10(c['label_fmt_fields']['dx_adv'])

    name = "dxadv_dxdiff_inv_" + name
    x_adv = PowerlawSeriesVariable('', 'dx_adv', dx_advs)
    xvar = PowerlawSeriesVariable('', 'invdelta', invdeltas)

    cache = PickleNodeCache('pickle', name)
    chain = chains.get_chain_powerlaw_datapoint(batch_cls, cache, np.inf, xparam_callback, additional_confine_ranges=[(1, 1, np.inf)])
    pms = PowerlawMultiSeries(chain, xvar, x_adv, cb, name=name, callback_kwargs={'param': param})
    nfig = NodeFigure(formats.powerlaws, title='Ratio of diffusive and advective step lengths', xlabel='$' + xlabel + '$')
    nfig.add(pms.datarows_chain)
    nfig.pad(.2)
    nfig[0].annotate('$\\Delta\\tau =' + str(param['dt']) + '$\n$\\sigma=0.5$', (0.50, 0.18), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
    nfig[0].legend(ncols=1, title='$\\delta^{{-1}}$')
    #nfig[0].format(xscale='log')
    nfig._legends_kw = {}
    nfig.savefig("figures/{}.pdf".format(name))
    return

    for val, ps in pms.series_dict.items():
        chain_x, chain_p = ps.histogram_chains
        nfig = NodeFigure(formats.doublehist, suptitle="Histograms for $\\Delta x_\\textrm{{adv}}={}$".format(val))
        if not spectra_xlim is None:
            def mod_h(h):
                dxadv = float(h.name.split('=')[1].split('_')[0])
                bw = (spectra_xlim[dxadv][1] - spectra_xlim[dxadv][0]) / 40
                h.set(bin_width=bw)
            chain_x.map_tree(mod_h, starts_with="histox")
            nfig[0].format(xlim=spectra_xlim[val])

        if not spectra_plim is None:
            nfig[1].format(xlim=spectra_plim)

        nfig.add(chain_x, 0, plot_on='spectra')
        nfig.add(chain_p, 1, plot_on='spectra')
        nfig.savefig("figures/{}_histograms_dxadv={}.pdf".format(name, val))

def fig_kruells_style(batch_cls, name, dx_advs, invdeltas, param, xaxis_delta=True, ylim=None):
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

        if 'sigma' in param:
            s = param['sigma']
            Xsh = (1 - s) * dx_adv + s * dx_diff
        else:
            Xsh = (dx_adv + dx_diff) / 2
        
        p = param | {'beta_s' : beta_s, 'q' : q, 'Xsh' : Xsh}
        label_p = p | {'dx_adv' : dx_adv, 'dx_diff' : dx_diff}
        return {'param': p, 'label_fmt_fields': label_p}

    xlabel = '\\delta^{-1} = \\Delta x_\\textrm{diff}/\\Delta x_\\textrm{adv}' if xaxis_delta else '\\Delta x_\\textrm{diff}'
    if xaxis_delta:
        xparam_callback = lambda c: c['label_fmt_fields']['dx_diff'] / c['label_fmt_fields']['dx_adv'] 
    else:
        xparam_callback = lambda c: c['label_fmt_fields']['dx_diff']

    name = "dxadv_dxdiff_" + name
    x_adv = PowerlawSeriesVariable('', 'dx_adv', dx_advs)
    xvar = PowerlawSeriesVariable(xlabel, 'invdelta', invdeltas)

    cache = PickleNodeCache('pickle', name)
    chain = chains.get_chain_powerlaw_datapoint(batch_cls, cache, np.inf, xparam_callback, additional_confine_ranges=[(1, 1, np.inf)])
    pms = PowerlawMultiSeries(chain, x_adv, xvar, cb, name=name, callback_kwargs={'param': param})
    nfig = NodeFigure(formats.powerlaws, suptitle='Ratio of diffusive and advective steps', xlabel='$' + xlabel + '$')
    nfig.add(pms.datarows_chain)
    nfig.pad(.2)
    nfig[0].annotate('$\\Delta\\tau =' + str(param['dt']) + '$\n$\\sigma=0.5$', (0.50, 0.18), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
    nfig[0].legend(ncols=1, title='$\\Delta x_\\textrm{adv}$')
    nfig[0].format(ylim=ylim)#, xlim=(0.8, 4.2))
    nfig._legends_kw = {}
    nfig.savefig("figures/{}.pdf".format(name))
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

    xlabel = '\\delta^{-1} = \\Delta x_\\textrm{diff}/\\Delta x_\\textrm{adv}' if xaxis_delta else '\\Delta x_\\textrm{diff}'
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
    x_adv = PowerlawSeriesVariable('\\Delta x_\\textrm{{adv}}', 'dx_adv', [0.0005, 0.05, 0.5, 5])
    xvar = PowerlawSeriesVariable(xlabel, 'invdelta', [1.01, 1.5, 2.0, 3.0, 3.5])

    cache = PickleNodeCache('pickle', name)
    chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells12, cache, 10, xparam_callback)
    pms = PowerlawMultiSeries(chain, x_adv, xvar, cb, name=name, callback_kwargs={'param': param})
    nfig = NodeFigure(formats.powerlaws, title='Influence of the ratio of step sizes on the power law index with eq. 19', xlabel='$' + xlabel + '$')
    nfig.add(pms.datarows_chain)
    nfig.pad(.2)
    nfig.savefig("figures/{}.pdf".format(name), legend_kw={'ncols': 1})
    pms.plot_histograms("figures", formats.doublehist)
    

param =  { 
      'r' : 4,
      't_inj' : 0.001,
      'x0' : 0,
      'y0' : 1,
      'k_syn' : 0,
      'Tmax' : 300,
      'dt':  0.05
    }

#fig_kruells_style(PyBatchKruells9, "9a", [0.0005, 0.005, 0.01, 0.05, 0.1, 0.5, 1], [1.01, 1.1, 1.2, 1.5, 1.8, 2.4, 3.0, 3.2, 3.4, 3.6], param)
#fig_kruells_style(PyBatchKruells9, "9a", [0.00005], [1.01, 1.1, 1.2, 1.5, 1.8, 2.4, 3.0, 3.2, 3.4, 3.6], param | {'t_inj' : 0.1})
fig_kruells_style(PyBatchKruells9, "9a", [0.005, 0.05, 0.1, 0.5, 1], [1.01, 1.1, 1.2, 1.5, 1.8, 2.4, 3.0, 3.2, 3.4, 3.6], param, ylim=(-4, -2.8))
#fig_kruells_style(PyBatchKruells14, "14", [0.0005, 0.05, 1.0], [1.01, 1.3, 1.8, 2.6, 3.4], param)
#fig_kruells_style(PyBatchKruells14, "14fast", [float(sys.argv[1])], [1.01, 1.3, 1.8, 2.6, 3.4], param | {'t_inj': 0.01, 'dt' : 0.1} ) ; exit()
#fig_kruells_style(PyBatchKruells14, "14fast", [0.000005, 0.00001, 0.00002, 0.00005, 0.0005, 0.005, 0.05, 1.0], [1.01, 1.3, 1.8, 2.6, 3.4], param | {'t_inj': 0.01, 'dt' : 0.1} )


""" the same stuff multiple times with different sigma """

spectra_xlim = {
        0.000005: (-0.01, 0.02), 
        0.00001: (-0.01, 0.04),
        0.00002: (-0.02, 0.05),
        0.00005: (-0.08, 0.18),
        0.0005: (-0.2, 0.4),
        0.005: (-2, 4),
        0.05: (-20, 40),
        1.0: (-200, 800)
    }
#fig_kruells_style_full(PyBatchKruells14, "14fast_sigma=0.8", [0.00001, 0.00002, 0.00005, 0.0005, 0.005, 0.05], [1.01, 1.3, 1.8, 2.6, 3.4], param | {'t_inj': 0.1, 'dt' : 0.1, 'sigma': 0.8}, spectra_xlim=spectra_xlim, spectra_plim=(1, 1e4))
#fig_kruells_style_full(PyBatchKruells14, "14fast_sigma=0.8", [float(sys.argv[1])], [1.01, 1.3, 1.8, 2.6, 3.4], param | {'t_inj': 0.1, 'dt' : 0.1, 'sigma': 0.8}, spectra_xlim=spectra_xlim, spectra_plim=(1, 1e4))

spectra_xlim = {
        0.000005: (-0.01, 0.02), 
        0.00001: (-0.01, 0.04),
        0.00002: (-0.02, 0.05),
        0.00005: (-0.08, 0.18),
        0.0005: (-0.2, 0.4),
        0.005: (-2, 4),
        0.05: (-20, 40),
        1.0: (-200, 800)
    }
# also already simulated: dxadv=0.000005 
#fig_kruells_style_full(PyBatchKruells14, "14fast_sigma=0.95", [0.00001, 0.00002, 0.00005, 0.0005, 0.005, 0.05], [1.01, 1.3, 1.8, 2.6, 3.4], param | {'t_inj': 0.01, 'dt' : 0.1, 'sigma': 0.95}, spectra_xlim=spectra_xlim, spectra_plim=(1, 1e4))
#fig_kruells_style(PyBatchKruells14, "14fast_sigma=0.95", [0.0005], [1.01, 1.3, 1.8, 2.6, 2.8, 2.9, 3.0, 3.2, 3.4], param | {'t_inj': 0.01, 'dt' : 0.1, 'sigma': 0.95} )
#fig_kruells_style(PyBatchKruells14, "14fast_sigma=0.95", [0.00005], [1.01, 1.3, 1.8, 2.0, 2.6, 2.9, 3.0, 3.1, 3.4], param | {'t_inj': 0.01, 'dt' : 0.1, 'sigma': 0.95} )

spectra_xlim = {
        0.000005: (-0.01, 0.02), 
        0.00001: (-0.01, 0.04),
        0.00002: (-0.02, 0.05),
        0.00005: (-0.08, 0.18),
        0.0005: (-0.2, 0.4),
        0.005: (-2, 4),
        0.05: (-20, 40),
        1.0: (-200, 800)
    }
#fig_kruells_style_full(PyBatchKruells14, "14fast", [0.000005, 0.00001, 0.00002, 0.00005, 0.0005, 0.005, 0.05, 1.0], [1.01, 1.3, 1.8, 2.6, 3.4], param | {'t_inj': 0.01, 'dt' : 0.1}, spectra_xlim=spectra_xlim )
#fig_kruells_style_full(PyBatchKruells14, "14fast", [0.00002, 0.00005, 0.0005, 0.005, 0.05], [1.01, 1.3, 1.8, 2.0, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 3.1, 3.4], param | {'t_inj': 0.01, 'dt' : 0.1}, spectra_xlim=spectra_xlim ) #also partially available: xadv=0.00001, 
#[2.2, 2.3, 2.4, 2.5, 2.8, 3.1][1.01, 1.3, 1.8, 2.6, 3.4]

# inverted datarow/x-axis
#fig_kruells_style_full_inv(PyBatchKruells14, "14fast", [0.00001, 0.00002, 0.00005, 0.0005, 0.005, 0.05], [1.01, 1.3, 1.8, 2.6, 3.4], param | {'t_inj': 0.01, 'dt' : 0.1}, spectra_xlim=spectra_xlim )


#fig_12()

