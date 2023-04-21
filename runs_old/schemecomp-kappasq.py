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

def cb(this_param, param):
        """
        Xdiff = 4sqrt(kappa dt)
        """
        dt = param['dt']
        dx_adv = param['dx_adv']
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

def schemecomp(dxadv):
        param = { 'Xsh' : 0.002,
              'beta_s' : 0.06,
              'r' : 4,
              'dt' : 0.001,
              't_inj' : 0.005, # should take ~6 min per invdelta on one core
              'k_syn' : 0,#.0001,
              'x0' : 0,
              'y0' : 1,
              'q' : 1,
              'Tmax' : 20.0
            }

        xlabel = '\\delta^{-1} = \\Delta x_\\textrm{diff}/\\Delta x_\\textrm{adv}' 
        xparam_callback = lambda c: c['label_fmt_fields']['dx_diff'] / c['label_fmt_fields']['dx_adv'] 
        invdeltas = [1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]
        if len(sys.argv) > 1:
                n = int(sys.argv[1])
                invdeltas = invdeltas[n * 2:n * 2 + 2]

        name = "schemecomp_kappasq_dx_adv=" + str(dxadv)
        xvar = PowerlawSeriesVariable(xlabel, 'invdelta', invdeltas)

        cache = PickleNodeCache('pickle', name)
        chain = chains.get_chain_powerlaw_datapoint(PyBatchKruells9, cache, np.inf, xparam_callback, additional_confine_ranges=[(1, 1, np.inf)])

        pls = PowerlawSeries(chain, xvar, cb, 
                        callback_kwargs={'param': param | {'dx_adv' : dxadv} }
                )

        axs_format = SliceDict()
        axs_format[2] = dict(xlabel='p/p_inj', xscale='log')
        axs_format[3] = dict(xlabel='p/p_inj', xscale='log')
        axs_format[4] = dict(xlabel='$\\epsilon$', ylabel='Powerlaw index $s$', xscale='log')
        axs_format[slice(None, None, None)] = dict(toplabels=('Spatial distribution', 'Momentum distribution', 'Powerlaw indizes'), leftlabels=('CES', 'KPPC'))
        nfmt = NodeFigureFormat(base=formats.histsandpowerlaw2, fig_legend_kw=None, axs_format=axs_format, legends_kw={0: None, 1: None, 2: None, 4:{}})
        nfig = NodeFigure(nfmt)
        nfig.format(suptitle='Fig. 2 of Achterberg/Schure')
        chain_x, chain_p = pls.histogram_chains

        datarow = pls.datarow_chain
        datarow.set(label='Cauchy-Euler')

        def mod_scheme(pls, batch_cls, name, name_human):
                memo = {}
                chain_x_old, chain_p_old = pls.histogram_chains
                datarow_old = pls.datarow_chain
                chain_x = chain_x_old.copy(name, last_kwargs={'batch_cls': batch_cls}, memo=memo)
                chain_p = chain_p_old.copy(name, last_kwargs={'batch_cls': batch_cls}, memo=memo)
                datarow = datarow_old.copy(name, last_parents={'batch_cls': batch_cls}, memo=memo)
                datarow.set(label=name_human)
                return datarow, chain_x, chain_p

        datarow_semiimplicit, chain_x_semiimplicit, chain_p_semiimplicit = mod_scheme(pls, PyBatchKruells14, 'semiimplicit', 'Semi-implicit')
        datarow_implicit, chain_x_implicit, chain_p_implicit = mod_scheme(pls, PyBatchKruells15, 'implicit', 'Implicit')
        datarow_kppc, chain_x_kppc, chain_p_kppc = mod_scheme(pls, PyBatchKruells16, 'kppc', 'Predictor-corrector')


        #nfig.add(chain_x, 0, plot_on='spectra')
        #nfig.add(chain_p, 2, plot_on='spectra')
        nfig.add(datarow, 4)

        #nfig.add(chain_x_kppc, 1, plot_on='spectra')
        #nfig.add(chain_p_kppc, 3, plot_on='spectra')
        nfig.add(datarow_kppc, 4)

        nfig.add(chain_x_implicit, 0, plot_on='spectra')
        nfig.add(chain_p_implicit, 2, plot_on='spectra')
        nfig.add(datarow_implicit, 4)

        nfig.add(chain_x_semiimplicit, 1, plot_on='spectra')
        nfig.add(chain_p_semiimplicit, 3, plot_on='spectra')
        nfig.add(datarow_semiimplicit, 4)

        nfig.pad(0.2, 4)
        nfig.savefig("figures/{}.pdf".format(name))

        nfig = NodeFigure(NodeFigureFormat(subplots={'ncols' : 1}, ax_format={'xscale': 'linear', 'xlabel': '$\\delta^{-1} = \\Delta x_\\textrm{diff}/\\Delta x_\\textrm{adv}$', 'ylabel': 'Power-law index'}, legend_kw={'ncols': 1, 'title': 'Scheme'}))
        #nfig[0].annotate('$\\Delta x_\\mathrm{{adv}} ={},~~\\sigma=0.5$'.format(dxadv), (0.21, 0.865), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
        nfig[0].annotate('\\hspace*{{-1.1cm}}$\\Delta x_\\mathrm{{adv}} ={}\\\\~~\\sigma=0.5$'.format(dxadv), (0.21, 0.53), xycoords='figure fraction', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
        nfig[0].annotate('Analytical expectation', (0.21, 0.89), xycoords='figure fraction', fontsize='x-small')#, bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=0.5))
        nfig[0].axhline(-2, color='red', lw=0.5)
        nfig.add(pls.datarow_chain)
        nfig.add(datarow_kppc)
        nfig.add(datarow_implicit)
        nfig.add(datarow_semiimplicit)
        nfig.pad(0.2)
        nfig.format(ylim=(-5.6, -1.8), suptitle='Comparison of numerical schemes')
        nfig.savefig("figures/{}-simple.pdf".format(name))

#schemecomp(0.00005)
schemecomp(0.1)
