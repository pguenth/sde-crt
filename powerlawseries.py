import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from node.special import *
from node.node import *
from node.cache import PickleNodeCache
from node.nodefigure import *
from pybatch.special.kruells import *
import proplot as pplt
import logging
import numpy as np
from dataclasses import dataclass

@dataclass
class PowerlawSeriesVariable:
    human_name: str
    name: str
    values: list

class PowerlawSeries:
    def __init__(self, chain, variable, last_kwargs_callback, name="", last_parents=None, other_values={}, callback_kwargs={}, label_template=None):
        """
        Creates a series of chains similar to chain with changed parameters.
        The chain is expected to have a parent called 'datapoint' which in turn
        is a NodeGroup (or similar) that can be used to fill a ScatterNode instance.

        All copies of the chain recieve last_kwargs as returned by the respective
        callback function and last_parents.

        If the chain also contains nodes named 'histox' and 'pl', the histograms
        of spatial and momentum distribution can also be plotted automatically
        (or manually by retrieving one NodeGroup per subplot).

        last_kwargs_callback: function(dict, **kwargs) -> dict
            is called once for each datapoint.
            the callback recieves a dict containing the variable name and value and
            items from other_values as well as items from callback_kwargs as kwargs

            it has to return a dict which is used as last_kwargs for the respective
            copy of the chain.
        """
        self.name = name
        self.var = variable
        self.last_kwargs_callback = last_kwargs_callback
        self.last_parents = last_parents
        self.other_values = other_values
        self.callback_kwargs = callback_kwargs

        if label_template is None:
            self.label_template = "Powerlaw indizes for different ${human_name}$"
        else:
            self.label_template = label_template

        self.chain = chain

        self._datarow_chain = None
        self._chains = None
        self._histogram_chains = None

    def _get_chains(self):
        d = {}
        for param_val in self.var.values:
            param_dict = self.other_values | {self.var.name: param_val}
            if self.name == "":
                new_name = "_{}={}".format(self.var.name, param_val)
            else:
                new_name = "_{}_{}={}".format(self.name, self.var.name, param_val)
            last_kwargs = self.last_kwargs_callback(param_dict, **self.callback_kwargs)
            d[param_val] = self.chain.copy(
                    new_name,
                    last_kwargs=last_kwargs,
                    last_parents=self.last_parents
                )

        return d

    @property
    def chains(self):
        if self._chains is None:
            self._chains = self._get_chains()

        return self._chains

    def _get_datarow_chain(self):
        label = self.label_template.format(human_name=self.var.human_name)
        chains = [dp.parents['datapoint'] for dp in self.chains.values()]
        return ScatterNode('scatter_{}'.format(self.name), chains, label=label, plot=True)

    @property
    def datarow_chain(self):
        if self._datarow_chain is None:
            self._datarow_chain = self._get_datarow_chain()

        return self._datarow_chain

    def _get_histogram_chains(self):
        xs = []
        ps = []
        for _, chain in self.chains.items():
            xs.append(chain.search_parent("histox"))
            ps.append(chain.search_parent("pl"))

        return NodeGroup("xs", xs), NodeGroup("ps", ps)

    @property
    def histogram_chains(self):
        if self._histogram_chains is None:
            self._histogram_chains = self._get_histogram_chains()

        return self._histogram_chains

    def plot_histograms(self, path, nfigure_format, title=None):
        chain_x, chain_p = self.histogram_chains
        nfig = NodeFigure(nfigure_format, suptitle=title)
        nfig.add(chain_x, 0, plot_on='spectra')
        nfig.add(chain_p, 1, plot_on='spectra')
        nfig.savefig(path)

    def plot_datarow(self, path, nfigure_format, **kwargs):
        nfig = NodeFigure(nfigure_format, **kwargs)
        nfig.add(self.datarow_chain)
        nfig.savefig(path)

class PowerlawMultiSeries:
    def __init__(self, chain, variable_rows, variable_points, last_kwargs_callback, name="", last_parents=None, other_values={}, callback_kwargs={}, label_template=None):
        self.name = name
        self.varr = variable_rows

        if label_template is None:
            label_template = "${var_name}={var_val}$"

        self.series_dict = {}
        for row_val in self.varr.values:
            label_t = label_template.format(var_name=self.varr.human_name, var_val=row_val)
            self.series_dict[row_val] = PowerlawSeries(
                    chain,
                    variable_points,
                    last_kwargs_callback,
                    name="{}={}".format(self.varr.name, row_val),
                    last_parents=last_parents,
                    other_values=other_values | {self.varr.name : row_val},
                    callback_kwargs=callback_kwargs,
                    label_template=label_t
                )

        self._datarows_chain = None
        self._histogram_chains = None

    def _get_datarows_chain(self):
        chains = [ps.datarow_chain for ps in self.series_dict.values()]
        return NodeGroup('multiseries', chains)

    @property
    def datarows_chain(self):
        if self._datarows_chain is None:
            self._datarows_chain = self._get_datarows_chain()

        return self._datarows_chain

    def _get_histogram_chains(self):
        return {val : ps.histogram_chains() for val, ps in self.series_dict.items()}

    @property
    def histogram_chains(self):
        if self._histogram_chains is None:
            self._histogram_chains = self._get_histogram_chains()

        return self._histogram_chains

    def plot_histograms(self, path, nfigure_format, title_template=None):
        if title_template is None:
            title_template = "Histograms for ${var_name}={var_val}$"

        for val, ps in self.series_dict.items():
            title = title_template.format(var_name=self.varr.human_name, var_val=val)
            if self.name == "":
                this_path = "{}/histograms_{}={}.pdf".format(path, self.varr.name, val)
            else:
                this_path = "{}/{}_histograms_{}={}.pdf".format(path, self.name, self.varr.name, val)
            ps.plot_histograms(this_path, nfigure_format, title)

    def plot_datarows(self, path, nfigure_format, **kwargs):
        nfig = NodeFigure(nfigure_format, **kwargs)
        nfig.add(self.datarows_chain)
        nfig.savefig(path)
