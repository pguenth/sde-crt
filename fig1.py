import sys
sys.path.insert(0, 'lib')
from src.specialnodes import *
from grapheval.node import *
from grapheval.cache import PickleNodeCache
from grapheval.nodefigure import *
from pybatch.special.kruells import *
import proplot as pplt
import logging
import numpy as np
import formats
import chains
from powerlawseries import *

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def acquire_chain(batch_cls, cache, confine_x, reeval, error_type='kiessling', bin_count=None):
    cycle = ColorCycle(['red', 'green', 'blue', 'yellow', 'black', 'violet'])
    datapoint = chains.get_chain_powerlaw_datapoint(
            batch_cls,
            cache,
            confine_x,
            xparam_callback=lambda c : c['batch_param']['Xsh'] / (c['batch_param']['beta_s'] * c['batch_param']['dt']),
            histo_opts={'label': '$X_\\textrm{{sh}}={Xsh}$', 'bin_count':bin_count},
            error_type=error_type
    )
    datapoint.search_parent("valuesp").ignore_cache = reeval
    datapoint.search_parent("valuesx").ignore_cache = reeval

    return datapoint

fig1_format = NodeFigureFormat(fig_format={
        'suptitle': "Power-law indices with constant diffusion coefficient",
        'xlabel': '$2^n=X_\\textrm{sh} / \\Delta x_\\textrm{adv}$',
        'ylabel': 'Power-law index $s$',
        'xscale': pplt.LogScale(base=2, subs=(1, 2)),
        'xmargin': 10
    })



def add_comparison(ax, comparison):
    labels = []
    has_label = True
    for xsh, s in comparison:
        ax.hlines(s, *(ax.get_xlim()), c='gray', ls='dotted', lw=1)#, label="Comparison" if has_label else None)
        labels.append("${}$".format(xsh))
        has_label = False

    ox = ax.dualy(lambda x : x)
    loc = np.array(comparison).T[1],
    ox.format(
        ylabel="$X_\\textrm{{sh}}$",
        ylocator=loc,
        yminorlocator=[],
        yticklabels=labels,
        yticklabelsize='x-small'
      )

def kw_callback(vars, **kwargs):
    param = kwargs['param']
    Xsh = 2**vars['n'] * vars['dxadv']
    dt = vars['dxadv'] / param['beta_s']
    d = {
            'param' : param | {'Xsh' : Xsh, 'dt' : dt},
            'label_fmt_fields' : {}
        }

    return d

def kw_callback_constdt(vars, **kwargs):
    param = kwargs['param']
    Xsh = 2**vars['n'] * vars['dxadv']
    beta_s = vars['dxadv'] / param['dt']
    d = {
            'param' : param | {'Xsh' : Xsh, 'beta_s' : beta_s},
            'label_fmt_fields' : {}
        }

    return d
                    

def fig1_def():
    param = { 
          'beta_s' : 1,
          'r' : 2,
          't_inj' : 0.1,
          'kappa' : 1,
          'x0' : 0,
          'y0' : 1,
          'L' : 10,
          'Tmax' : 300
        }

    x_advs = [0.256, 0.128, 0.064, 0.032, 0.016, 0.008]
    ns = [-2, -1, 0, 1, 2, 3]

    kruells94_comparison = [(0.001, -4), (0.064, -4.12), (0.128, -4.216), (0.256, -4.408), (0.512, -4.8), (1.024, -5.58)]

    name = 'fig1_def'
    cache = PickleNodeCache('pickle', name)
    datapoint = acquire_chain(PyBatchKruells10, cache, np.inf, False, 'linregress')

    rows = PowerlawSeriesVariable("", "dxadv", x_advs)
    points = PowerlawSeriesVariable("n", "n", ns)
    pms = PowerlawMultiSeries(datapoint, rows, points, kw_callback, name=name, callback_kwargs={'param': param})

    nfig = NodeFigure(fig1_format, xlocator=2.0**np.array(ns))
    nfig.add(pms.datarows_chain, instant=True)
    nfig.pad(0.2)
    add_comparison(nfig[0], kruells94_comparison)

    nfig[0].legend(title="$\\Delta x_\\textrm{adv}$", ncols=2, order='F')
    nfig.savefig('figures/{}.pdf'.format(name))

    #pms.plot_histograms('figures', formats.doublehist)

def fig1_ext():
    param = { 
          'dt' : 0.1,
          'r' : 2,
          't_inj' : 0.1,
          'kappa' : 10,
          'x0' : 0,
          'y0' : 1,
          'L' : 10,
          'Tmax' : 300
        }

    x_advs = [0.256, 0.128, 0.064, 0.032, 0.016, 0.008]
    ns = [-4,-3, -2, -1, 0, 1, 2, 3]

    kruells94_comparison = [(0.001, -4), (0.064, -4.12), (0.128, -4.216), (0.256, -4.408), (0.512, -4.8), (1.024, -5.58)]

    name = 'fig1_ext'
    cache = PickleNodeCache('pickle', name)
    datapoint = acquire_chain(PyBatchKruells10, cache, np.inf, True, 'kiessling', bin_count=60)

    rows = PowerlawSeriesVariable("", "dxadv", x_advs)
    points = PowerlawSeriesVariable("n", "n", ns)
    pms = PowerlawMultiSeries(datapoint, rows, points, kw_callback_constdt, name=name, callback_kwargs={'param': param})

    nfig = NodeFigure(fig1_format, xlocator=2.0**np.array(ns), title="Momentum power law indices (constant diffusion coefficient)")
    nfig.add(pms.datarows_chain, instant=True)
    nfig.pad(0.2)
    add_comparison(nfig[0], kruells94_comparison)

    nfig[0].legend(title="$\\Delta x_\\textrm{adv}$", ncols=2, order='F')
    nfig.savefig('figures/{}.pdf'.format(name))

    pms.plot_histograms('figures', formats.doublehist)

def fig1_9a():
    param = { 
          'beta_s' : 0.08,
          'r' : 4,
          't_inj' : 0.001,
          'x0' : 0,
          'y0' : 1,
          'q' : 2,
          'k_syn' : 0,
          'Tmax' : 300
        }

    x_advs = [0.256, 0.128, 0.064, 0.032, 0.016, 0.008]
    ns = [-2, -1, 0, 1, 2, 3]

    name = 'fig1_9a'
    cache = PickleNodeCache('pickle', name)
    datapoint = acquire_chain(PyBatchKruells9, cache, np.inf, False)

    rows = PowerlawSeriesVariable("\\Delta x_\\textrm{{adv}}", "dxadv", x_advs)
    points = PowerlawSeriesVariable("n", "n", ns)
    pms = PowerlawMultiSeries(datapoint, rows, points, kw_callback, name=name, callback_kwargs={'param': param})

    nfig = NodeFigure(fig1_format, xlocator=2.0**np.array(ns))
    nfig.add(pms.datarows_chain, instant=True)
    nfig.pad(0.2)
    add_comparison(nfig[0], kruells94_comparison)

    nfig.savefig('figures/{}.pdf'.format(name))

    pms.plot_histograms('figures', formats.doublehist)


fig1_def()
