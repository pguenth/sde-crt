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
from dataclasses import dataclass
from typing import Callable

def fa_with(**kwargs):
    fig = pplt.figure()
    ax = fig.subplot()
    ax.format(**kwargs)

    return fig, ax

class PowerlawSeries:
    def __init__(self, name, batch_cls, param_name, param_values, def_params, abstract_param={}, param_callback=None, confine_x=np.inf, reeval=False, param_human_name=None):
        """
        param_callback: function(dict, dict) -> dict
            is called once for each datapoint.
            the first dict passed is a dictionary containing the values of the parameters
            that are changed in the simulation, the second dict is the dict of default
            parameters and the third dict is abstract_param.

            has to return a dict with simulation parameters that have to be modified for 
            the respective run and their values and a second dict with other parameters
            that are passed on to label_fmt_fields
        """
        self.name = name
        self.param_name = param_name
        self.param_values = param_values
        self.def_params = def_params
        self.abstract_param = abstract_param
        self._chains = None
        self._datarow_chain = None
        if param_human_name is None:
            self.param_human_name = self.param_name
        else:
            self.param_human_name = param_human_name

        self.confine_x = confine_x

        self.cachepath = "pickle"
        self.figpath = "figures"

        if param_callback is None:
            # simply return the first dict
            self.param_callback = lambda p, d, a: (p, a)
        else:
            self.param_callback = param_callback

        self._acquire_datapoint_chain(batch_cls, reeval)

    def _get_chains(self):
        d = {}
        d_spatial = {}
        for param_val in self.param_values:
            param_dict = {self.param_name: param_val}
            new_name = "_{}={}".format(self.param_name, param_val)
            modified_params, additional_params = self.param_callback(param_dict, self.def_params, self.abstract_param)
            d[param_val] = self.chain.copy(
                    new_name,
                    param=self.def_params | modified_params,
                    label_fmt_fields=modified_params | additional_params
                )
            d_spatial[param_val] = self.chain_spatial.copy( 
                    new_name,
                    last_parents={'points': d[param_val].search_parent('points')}
                )

        return d, d_spatial

    @property
    def chains_spatial(self):
        if self._chains is None:
            self._chains = self._get_chains()

        return self._chains[1]

    @property
    def chains(self):
        if self._chains is None:
            self._chains = self._get_chains()

        return self._chains[0]

    def _get_datarow_chain(self):
        label = "Powerlaw indizes for different ${}$".format(self.param_human_name)
        return ScatterNode('scatter', self.chains, label=label, plot=True)

    @property
    def datarow_chain(self):
        if self._datarow_chain is None:
            self._datarow_chain = self._get_datarow_chain()

        return self._datarow_chain

    def _acquire_datapoint_chain(self, batch_cls, reeval):
        """
        Get the evaluation chain for one data point
        """
        cycle = ColorCycle(['red', 'green', 'blue', 'yellow', 'black', 'violet'])
        cache = PickleNodeCache(self.cachepath, self.name)

        batch = BatchNode('batch',
                batch_cls = batch_cls,
                cache=cache,
                ignore_cache=False
            )

        points = PointNode('points', {'batch' : batch}, cache=cache, ignore_cache=False)

        valuesp = ValuesNode('valuesp', 
                {'points' : points},
                index=1,
                cache=cache, 
                ignore_cache=reeval,
                confinements=[(0, lambda x : np.abs(x) < self.confine_x)]
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
                label="${}={{{}}}$".format(self.param_human_name, self.param_name)
            )

        powerlaw = PowerlawNode(
                'pl', 
                {'dataset' : histogramp },
                plot=False,
                color_cycle=cycle
            )

        xparam_get = CommonCallbackNode(
                'xparam_get',
                parents=histogramp,
                callback=lambda c: c['batch_param'][self.param_name]
            )

        self.chain = NodeGroup('group', {'x' : xparam_get, 'y': powerlaw[1], 'dy' : powerlaw[3]})

        valuesx = ValuesNode('valuesx', 
                index=0,
                cache=cache, 
                ignore_cache=False,
            )

        histogramx = HistogramNode('histox',
                {'values' : valuesx}, 
                bin_count=40, 
                normalize='width',
                log_bins=False, 
                plot=True,
                cache=cache,
                ignore_cache=False,
                style='line',
            )

        self.chain_spatial = histogramx  

    def get_histograms(self, title=None):
        fig, axs = pplt.subplots(ncols=2, share=False, tight=True)
        axs[1].format(
            xscale='log',
            yscale='log',
            xformatter=pplt.SciFormatter(),
            yformatter=pplt.SciFormatter(),
            title=title,
            xlabel='$p/p_\\textrm{inj}$',
            ylabel='particle number density'
        )

        axs[0].format(yscale='log', xlabel='x', ylabel='particle number density')

        for _, chain in self.chains.items():
            hist = chain.search_parent("histop")
            pl = chain.search_parent("pl")

            hist.do_plot = True
            pl.do_plot = True

            pl(axs[1])

            hist.do_plot = False
            pl.do_plot = False

        for _, node_chain in self.chains_spatial.items():
            node_chain(axs[0])

        axs[1].legend(loc='r', ncol=1)
        fig.savefig("{}/{}-histograms.pdf".format(self.figpath, self.name))

    def get_series(self, title=None):
        fig, ax = fa_with(
            xscale='log',
            xformatter=pplt.SciFormatter(),
            yformatter=pplt.SciFormatter(),
            title=title,
            xlabel="${}$".format(self.param_human_name),
            ylabel='Powerlaw index'
        )

        self.datarow_chain(ax)

        ax.legend(ncol=1)
        fig.savefig("{}/{}-series.pdf".format(self.figpath, self.name))

@dataclass
class PowerlawStudyConfig:
    datarows_name: str
    # one datarow is generated for each value in this list
    # when calling plot_momentum_spectra, one plot is created
    # for each value in this list each containing one histogram
    # for each value in datapoint_values
    datarow_values: list
    datarow_label: str

    datapoints_name: str

    # the values that are passed one at a time to param_callback as second parameter
    datapoint_values: list

    # the label for the momentum spectra histograms.
    # .format is called on this label with the values of both dicts returned by param_callback
    datapoint_label: str 

    xlabel: str
    xscale: str

    datapoint_id_fmt_str: str

    # the default parameters passed to each pybatch instance (see param_callback)
    def_param: dict

    # parameters that param_callback needs for additional calculations
    add_param: dict

    confine_x: float

    # param_callback is passed:
    #   - one value of datarow_values 
    #   - one value of datapoint_values
    #   - def_param
    #   - add_param 
    # It is expected to return two dicts, one passed to the pybatch instance
    # as parameters (in addition to def_param) and one that is passed to label_fmt_fields
    # and to other String.format calls
    param_callback: Callable

    # this callable is passed to the CommonCallbackNode used to generate the x-axis values
    # from the common dict (which is containing the return value of param_callback as 
    # label_fmt_fields and batch_param resp).
    xparam_callback: Callable

    figpath: str = "figures"
    cachepath: str = "pickle"


class PowerlawStudy:
    def __init__(self, name, batch_cls, config):
        self.name = name

        self.datarows_name = config.datarows_name
        self.datarow_values = config.datarow_values
        self.datarow_label = config.datarow_label

        self.datapoints_name = config.datapoints_name
        self.datapoint_values = config.datapoint_values
        self.datapoint_label = config.datapoint_label

        self.datapoint_id_fmt_str = config.datapoint_id_fmt_str

        self.xlabel = config.xlabel
        self.xscale= config.xscale

        self.def_param = config.def_param
        self.add_param = config.add_param

        self.confine_x = config.confine_x

        self.param_callback = config.param_callback
        self.xparam_callback = config.xparam_callback

        self.figpath = config.figpath
        self.cachepath = config.cachepath

        self._acquire_datapoint_chain(batch_cls)
        self._datapoint_chains = None
        self._datarow_chains = None
        self._full_chain = None
    
    def _get_datapoint_chains(self):
        dr_dict = {}
        dr_dict_spatial = {}
        for dr_val in self.datarow_values:
            d = {}
            d_spatial = {}
            for dp_val in self.datapoint_values:
                new_param, new_meta_param = self.param_callback(dr_val, dp_val, self.def_param, self.add_param)
                new_id = ('_' + self.datapoint_id_fmt_str).format(**(new_param | new_meta_param))
                d[dp_val] = self.datapoint_chain.copy(
                        new_id,
                        param = self.def_param | new_param,
                        label_fmt_fields = new_param | new_meta_param
                    )
                d_spatial[dp_val] = self.datapoint_chain_spatial.copy(
                        new_id,
                        last_parents={'points': d[dp_val].search_parent('points')}
                    )
            dr_dict[dr_val] = d
            dr_dict_spatial[dr_val] = d_spatial

        return dr_dict, dr_dict_spatial

    def _get_datarow_chains(self):
        datarow_chains = []
        for dr_val, dp_chain in self.datapoint_chains.items():
            n = ScatterNode('scatter_{}={}'.format(self.datarows_name, dr_val), dp_chain, label=self.datarow_label.format(dr_val), plot=True)
            datarow_chains.append(n)

        return datarow_chains

    @property
    def datapoint_chains(self):
        if self._datapoint_chains is None:
            self._datapoint_chains = self._get_datapoint_chains()

        return self._datapoint_chains[0]

    @property
    def datapoint_chains_spatial(self):
        if self._datapoint_chains is None:
            self._datapoint_chains = self._get_datapoint_chains()

        return self._datapoint_chains[1]

    @property
    def datarow_chains(self):
        if self._datarow_chains is None:
            self._datarow_chains = self._get_datarow_chains()

        return self._datarow_chains

    @property
    def full_chain(self):
        if self._full_chain is None:
            self._full_chain = NodeGroup('datarows', self.datarow_chains)

        return self._full_chain

    def plot_momentum_spectra(self, title_fmt_str=None):
        if title_fmt_str is None:
            title_fmt_str = 'Power laws for different diffusion steps. ${}={}$'

        for dr_val in self.datapoint_chains.keys():
            fig, axs = pplt.subplots(ncols=2, share=False, tight=True)
            axs[1].format(
                xscale='log',
                yscale='log',
                xformatter=pplt.SciFormatter(),
                yformatter=pplt.SciFormatter(),
                title=title_fmt_str.format(self.datarows_name, dr_val),
                xlabel='$p/p_\\textrm{inj}$',
                ylabel='particle number density'
            )

            axs[0].format(yscale='log', xlabel='x', ylabel='particle number density')

            for _, node_chain  in self.datapoint_chains[dr_val].items():
                hist = node_chain.search_parent("histop")
                pl = node_chain.search_parent("pl")

                hist.do_plot = True
                pl.do_plot = True

                pl(axs[1])

            for _, node_chain  in self.datapoint_chains_spatial[dr_val].items():
                node_chain(axs[0])

            axs[1].legend(loc='r', ncol=1)
            fig.savefig("{}/{}-{}={}.pdf".format(self.figpath, self.name, self.datarows_name, dr_val))

    def _acquire_datapoint_chain(self, batch_cls):
        """
        Get the evaluation chain for one data point
        """
        cycle = ColorCycle(['red', 'green', 'blue', 'yellow', 'black', 'violet'])
        cache = PickleNodeCache(self.cachepath, self.name)

        batch = BatchNode('batch',
            batch_cls = batch_cls,
            cache=cache,
            ignore_cache=False
            )

        points = PointNode('points', {'batch' : batch}, cache=cache, ignore_cache=False)

        valuesp = ValuesNode('valuesp', 
                {'points' : points},
                index=1,
                cache=cache, 
                ignore_cache=False,
                confinements=[(0, lambda x : np.abs(x) < self.confine_x)]
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
                label=self.datapoint_label
            )

        powerlaw = PowerlawNode(
                'pl', 
                {'dataset' : histogramp },
                plot=False,
                color_cycle=cycle
            )

        xparam_get = CommonCallbackNode(
                'xparam_get',
                parents=histogramp,
                callback=self.xparam_callback
            )

        self.datapoint_chain = NodeGroup('group', {'x' : xparam_get, 'y': powerlaw[1], 'dy' : powerlaw[3]})

        valuesx = ValuesNode('valuesx', 
                index=0,
                cache=cache, 
                ignore_cache=False,
            )

        histogramx = HistogramNode('histox',
                {'values' : valuesx}, 
                bin_count=15, 
                normalize='width',
                log_bins=False, 
                plot=True,
                cache=cache,
                ignore_cache=False,
                style='line',
            )

        self.datapoint_chain_spatial = histogramx  

    def plot_datarows(self, title=None, comparison=None, comparison_label=None):
        if title is None:
            title = "Momentum spectrum power law indices for\\\\different diffusion and advection lengths"

        fig = pplt.figure(suptitle=title, tight=True)
        ax = fig.subplot()
        ax.format(
            xlabel=self.xlabel,
            ylabel='Powerlaw index $s$',
            xscale=self.xscale,
            #xscale=pplt.LogScale(base=2, subs=(1, 2)),
            #xlocator=invdeltas,
            xmargin=10
        )


        self.full_chain(ax)

        xlim = ax.get_xlim()
        ax.format(xlim=(xlim[0] * 0.8, xlim[1] * 1.2))

        if not comparison is None and not comparison_label is None:
            labels = []
            for comp_val, s in comparison:
                ax.hlines(s, *(ax.get_xlim()), c='gray', ls='dotted', lw=1)
                labels.insert(0, "${}$".format(comp_val))

            ox = ax.dualy(lambda x : x)
            ox.format(
                ylabel=comparison_label,
                ylocator=np.array(comparison).T[1],
                yminorlocator=[],
                yticklabels=labels,
                yticklabelsize='x-small'
              )


        ax.legend(loc='ll', ncol=1)

        fig.savefig('{}/{}.pdf'.format(self.figpath, self.name))
