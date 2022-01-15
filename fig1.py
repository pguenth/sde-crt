import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from node.special import *
from node.node import *
from node.cache import PickleNodeCache
from pybatch.special.kruells import *
import proplot as pplt
import logging
import numpy as np

pplt.rc.update({
        'text.usetex' : True,
        })

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def fa_with(**kwargs):
    fig = pplt.figure()
    ax = fig.subplot()
    ax.format(**kwargs)

    return fig, ax

def plot_momentum_spectra(d, name, path="figures"):
    for xadv in d.keys():
        fig, ax = fa_with(
            xscale='log',
            yscale='log',
            xformatter=pplt.SciFormatter(),
            yformatter=pplt.SciFormatter(),
            title='Power laws for different shock widths. $x_\\textrm{{adv}}={}$'.format(xadv),
            xlabel='$p/p_\\textrm{inj}$',
            ylabel='particle number density'
        )

        for xsh, node_chain  in d[xadv].items():
            hist = node_chain.search_parent("histop")
            pl = node_chain.search_parent("pl")

            memo = hist(ax, plot_this=True)
            pl(ax, plot_this=True, memo=memo)

        ax.legend(loc='r', ncol=1)
        fig.savefig("{}/{}-xadv={}.pdf".format(path, name, xadv))

def generate_xsh_xadv_range(node_chain, param, x_advs, ns):
    x_adv_d = {}
    beta_s = param['beta_s']
    for x_adv in x_advs:
        d = {}
        dt = x_adv / beta_s
        for n in ns:
            Xsh = 2**n * x_adv
            d[Xsh] = node_chain.copy(
                    '_xadv={}_Xsh={}'.format(x_adv, Xsh),
                    param = param | {'Xsh' : Xsh, 'dt' : dt}
                )
        x_adv_d[x_adv] = d

    return x_adv_d

def acquire_chain(name, batch_cls, x_advs, ns, param):
    cycle = ColorCycle(['red', 'green', 'blue', 'yellow', 'black', 'violet'])
    cache = PickleNodeCache('pickle', name)

    batch = BatchNode('batch',
        batch_cls = batch_cls,
        cache=cache,
        ignore_cache=False
        )

    points = PointNode('points', {'batch': batch}, cache=cache, ignore_cache=False)

    valuesp = ValuesNode('valuesp', 
            {'points' : points},
            index=1,
            cache=cache, 
            ignore_cache=False,
            confinements=[(0, lambda x : np.abs(x) < 0.1)]
        )

    histogramp = HistogramNode('histop',
            {'values' : valuesp}, 
            bin_count=15, 
            normalize='density',
            log_bins=True, 
            plot=False,
            cache=cache,
            ignore_cache=False,
            style='line',
            color_cycle=cycle,
            label='$X_\\textrm{{sh}}={Xsh}$'
        )

    powerlaw = PowerlawNode(
            'pl', 
            {'dataset' : histogramp },
            plot=False,
            color_cycle=cycle
        )

    n_get = CommonCallbackNode(
            'nget',
            parents=histogramp,
            callback=lambda c : c['batch_param']['Xsh'] / (c['batch_param']['beta_s'] * c['batch_param']['dt'])
        )

    fig1_datapoint = NodeGroup('group', {'x' : n_get, 'y': powerlaw[1], 'dy' : powerlaw[3]})

    datapoint_nodes = generate_xsh_xadv_range(fig1_datapoint, param, x_advs, ns) 
    datarow_nodes = []
    for xadv, dp_nodes in datapoint_nodes.items():
        n = ScatterNode('scatter_xadv={}'.format(xadv), dp_nodes, label='$\\Delta x_\\textrm{{adv}}={}$'.format(xadv), plot=True)
        datarow_nodes.append(n)

    fig1node = NodeGroup('datarows', datarow_nodes)
    return fig1node, datapoint_nodes

def fig1_plot(chain, name, ns, path="figures", title=None, comparison=None):
    if title is None:
        title = "Momentum spectrum power law indices for\\\\different shock widths and advection lengths"

    fig = pplt.figure(suptitle=title, tight=True)
    ax = fig.subplot()
    ax.format(
        xlabel='$2^n=X_\\textrm{sh} / \\Delta x_\\textrm{adv}$',
        ylabel='Powerlaw index $s$',
        xscale=pplt.LogScale(base=2, subs=(1, 2)),
        xlocator=2.0**np.array(ns),
        xmargin=10
    )


    chain(ax)

    xlim = ax.get_xlim()
    ax.format(xlim=(xlim[0] * 0.8, xlim[1] * 1.2))

    if not comparison is None:
        labels = []
        for xsh, s in comparison:
            ax.hlines(s, *(ax.get_xlim()), c='gray', ls='dotted', lw=1)
            labels.insert(0, "${}$".format(xsh))

        ox = ax.dualy(lambda x : x)
        ox.format(
            ylabel="$X_\\textrm{{sh}}$",
            ylocator=np.array(comparison).T[1],
            yminorlocator=[],
            yticklabels=labels,
            yticklabelsize='x-small'
          )


    ax.legend(loc='ll', ncol=1)

    fig.savefig('{}/{}.pdf'.format(path, name))

def fig1_def():
    param = { 
          'beta_s' : 1,
          'r' : 2,
          't_inj' : 0.001,
          'kappa' : 1,
          'x0' : 0,
          'y0' : 1,
          'L' : 10,
          'Tmax' : 300
        }

    x_advs = [0.256, 0.128, 0.064, 0.032, 0.016, 0.008]
    ns = [-2, -1, 0, 1, 2, 3]

    kruells94_comparison = [(0.001, -4), (0.064, -4.12), (0.128, -4.216), (0.256, -4.408), (0.512, -4.8), (1.024, -5.58)]

    fig1node, datapoint_nodes = acquire_chain('fig1_def', PyBatchKruells10, x_advs, ns, param)
    fig1_plot(fig1node, 'fig1_def', ns, comparison=kruells94_comparison)
    plot_momentum_spectra(datapoint_nodes, 'fig1_def')

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


    fig1node, datapoint_nodes = acquire_chain('fig1_9a', PyBatchKruells9, x_advs, ns, param)
    fig1_plot(fig1node, 'fig1_9a', ns)
    plot_momentum_spectra(datapoint_nodes, 'fig1_9a')

fig1_9a()
