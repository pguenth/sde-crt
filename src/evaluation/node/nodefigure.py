from collections.abc import MutableMapping
import proplot as pplt
import copy
from .node import NodeGroup
from .graph import draw_node_chain

pplt.rc.update({
                'text.usetex' : True,
                })

class NoLegend:
    pass

class SliceDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        """
        A MutableMapping subclass (i.e. dict) which accepts 
        slices as keys by reducing them
        """
        self._d = {}
        if len(args) == 1 and type(args[0]) is type(self):
            self.update(args[0])
            self.update(dict(**kwargs))
        else:
            self.update(dict(*args, **kwargs))

    @staticmethod
    def _transform_key(key):
        if isinstance(key, slice):
            key = key.__reduce__()
        return key

    def __getitem__(self, key):
        return self._d[self._transform_key(key)]

    def __setitem__(self, key, value):
        self._d[self._transform_key(key)] = value

    def __delitem__(self, key):
        del self._d[self._transform_key(key)]

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        for k in self._d:
            if isinstance(k, tuple):
                if k[0] is slice:
                    yield slice(*k[1])
                else:
                    yield k
            else:
                yield k

    def __str__(self):
        s = ""
        for k, v in self.items():
            s += "{}: {}, ".format(k, v)

        if s == "":
            return "{}"
        else:
            return "{" + s[:-2] + "}"



class NodeFigureFormat:
    def __init__(self, base=None, subplots=None, fig_format=None, axs_format=None, ax_format=None, legend_kw=None, legends_kw=None, fig_legend_kw=None):
        """
        A set of format options for repeated use in NodeFigure instances.
        
        base: another NodeFigureFormat instance which represents the base of this instance
        subplots: dict of args passed to Figure.add_subplots()
        fig_format: dict of args passed to Figure.format() and therefore may also apply
            to all axes globally
        ax_format: dict of args passed to SubplotGrid.format(), i.e. all Axes objects
            is ignored if axs_format is not None
        axs_format:
            * list of dicts with args passed to every subplot's Axes.format()
              The SubplotGrid is seen as flattened array.
            or
            * dict of dicts with args passed to every subplot Axes.format().
              keys of the dict must be valid keys for the .__getitem__() method of
              the SubplotGrid instance returned with the given subplots parameters
              For example, if you set subplots to create a 3x3 grid, you can use all
              keys that yield items inside of this 3x3 grid.
              If you want to use slices as keys you may use the provided SliceDict class.
        legend_kw: dict of args passed to Axes.legend() for all Axes objects. Ignored
            if legends_kw is set.
        legends_kw: 
            * list of dicts with args passed to every subplot's Axes.legend()
              The SubplotGrid is seen as flattened array.
            or
            * dict of dicts with args passed to every subplot Axes.legend().
              keys of the dict must be valid keys for the .__getitem__() method of
              the SubplotGrid instance returned with the given subplots parameters
              For example, if you set subplots to create a 3x3 grid, you can use all
              keys that yield items inside of this 3x3 grid.
              If you want to use slices as keys you may use the provided SliceDict class.
        """
        if base is None:
            if subplots is None:
                self.subplots = {'ncols' : 1, 'nrows' : 1}
            else:
                self.subplots = subplots

            if fig_format is None:
                self.fig_format = {}
            else:
                self.fig_format = fig_format

            self.fig_legend_kw = fig_legend_kw

            self.axs_format = NodeFigureFormat._multi_param_parse(axs_format, ax_format)
            self.legends_kw = NodeFigureFormat._multi_param_parse(legends_kw, legend_kw)
        else:
            if subplots is None:
                self.subplots = base.subplots
            else:
                self.subplots = base.subplots | subplots

            if fig_format is None:
                self.fig_format = base.fig_format
            else:
                self.fig_format = base.fig_format | fig_format

            if fig_legend_kw is None:
                self.fig_legend_kw = base.fig_legend_kw
            elif base.fig_legend_kw is None:
                self.fig_legend_kw = fig_legend_kw
            else:
                self.fig_legend_kw = base.fig_legend_kw | fig_legend_kw

            axs_format_extend = NodeFigureFormat._multi_param_parse(axs_format, ax_format)
            legends_kw_extend = NodeFigureFormat._multi_param_parse(legends_kw, legend_kw)
            
            self.axs_format = NodeFigureFormat._slice_dict_merge(base.axs_format, axs_format_extend)
            self.legends_kw = NodeFigureFormat._slice_dict_merge(base.legends_kw, legends_kw_extend)

    def __getitem__(self, key):
        return self.axs_format[key]

    def __repr__(self):
        return "NodeFigureFormat({}, {}, axs_format={}, legends_kw={})".format(self.subplots, self.fig_format, self.axs_format, self.legends_kw)

    def __str__(self):
        return repr(self)

    @staticmethod
    def _slice_dict_merge(base, extend):
        """
        Merges every k:v pair of extend into the respective kb:vb pair in base
        if the latter exists (vb |= v is used). Else add k:v to base
        """
        new = copy.deepcopy(base)
        for extend_k, extend_d in extend.items():
            if extend_d is None and extend_k in new:
                del new[extend_k]
            elif extend_k in new:
                new[extend_k] |= extend_d
            else:
                new[extend_k] = extend_d

        return new


    @staticmethod
    def _multi_param_parse(specific, dict_all):
        """
        Returns a SliceDict (SubplotGrid key -> dict) for any possible
        choise of specific and dict_all.

        specific may be either a list in which case it is enumerated
        or a dict (which is returned unchanged)

        if specific is None, dict_all is returned with a key
        representing all SubplotGrid members
        """
        if specific is None:
            if dict_all is None:
                return SliceDict()
            else:
                s = SliceDict()
                s[slice(None, None, None)] = dict_all
                return s
        elif isinstance(specific, list):
            return SliceDict(enumerate(specific))
        elif isinstance(specific, MutableMapping):
            return SliceDict(specific)

        raise ValueError("Specific must be either list or dict or None")

class NodeFigure:
    def __init__(self, default_format, **kwargs):
        """
        proplot.Figure wrapper for easy default-formatting of plots
        and addition of node_chains.
        Workflow: Create a NodeFigureFormat instance for each type of plot
        that you are about to create and pass it to a NodeFigure instance
        where needed. Change global parameters in the NodeFigure constructor
        and Subplot-special parameters by using NodeFigure[index].format()

        default_format: an instance of NodeFigureFormat containing the defaults
        **kwargs: kwargs passed to Figure.format() and subsequently to all Axes.format()
        """
        self._create(default_format)
        self.format(**kwargs)
        self._legends_kw = default_format.legends_kw

        # atm only one legend possible, maybe use list for multiple calls or
        # use a dict {span : kw} 
        self._fig_legend_kw = default_format.fig_legend_kw

        # _chains contains the node_chains of this figure as keys and
        # a tuple of
        #   * the respective key for SubplotGrid.__getitem() to be executed on
        #   * a boolean that is True if this chain has been executed
        #   * a dict with kwargs passed to the node chain __call__ method
        # as values
        self._chains = {}

        # contains SubplotGrid keys as keys and keeps a memo dict as value
        # for each key. This prevents multiple plotting on the same set
        # of subfigures if two chain entrypoints have the same parents
        # at some point. It does not prevent mutltiple plotting on overlapping
        # slices that are given, this may be implemented in the future
        self._memos = SliceDict()

    def __getitem__(self, key):
        return self.axs[key]

    @property
    def axs(self):
        return self._axs

    @property
    def fig(self):
        return self._fig
    
    @property
    def figure(self):
        return self.fig

    def format(self, *args, **kwargs):
        self.figure.format(*args, **kwargs)

    def execute(self):
        """
        Execute all remaining node_chains now.
        """
        for chain, (key, ran, kwargs) in self._chains.items():
            if not ran:
                self._execute_one_chain(chain, key, **kwargs)

    def add(self, node_chain, to=None, instant=True, plot_on=None, memo=None, **kwargs):
        """
        Add a node_chain to the subplot given by to.
        If to is None (default) the node chain is added
        to every subplot in this figure. This is most useful for figures
        with only one subplot.
        Else to is interpreted as a slice for the SubplotGrid.__getitem__()
        method and the node_chain is added to all subplots returned by
        this call.

        If instant is set to True the node_chain is executed
        in this function. If it is set to False the node_chain
        is scheduled to execute when calling savefig.

        plot_this and memo are passed to the node chain, as well as all
        other kwargs. If memo is None the NodeFigure instance uses a memo
        list which is the same for every call with the same value of *to*.
        """
        if to is None:
            to = slice(None, None, None)

        if memo is None:
            if not to in self._memos:
                self._memos[to] = list()
            memo = self._memos[to]

        node_args = {'plot_on' : plot_on, 'memo' : memo} | kwargs

        if instant:
            rval = self._execute_one_chain(node_chain, to, **node_args)
        else:
            rval = None

        self._chains[node_chain] = (to, instant, node_args)

        return rval

    def savefig(self, path, legend_kw=None, legends_kw=None, fig_legend_kw=None, savefig_args=None):
        """
        Save figure to path (using savefig_args as kwargs for savefig)
        after adding a legend where applicable.

        Node chains that have been added but not run are executed now. 

        legend_kw and legends_kw work exactly like NodeFigureFormat's equally
        named parameters. If one is set for this method, NodeFigureFormat's 
        parameters are both overridden. See in NodeFigureFormat for more
        information. WARNING: At the moment those two parameters are ignored

        """
        if savefig_args is None:
            savefig_args = {}

        self.execute()

        #if not legend_kw is None or not legends_kw is None:
        #    legends_kw_extend = NodeFigureFormat._multi_param_parse(legends_kw, legend_kw)
        #    legends_kw = NodeFigureFormat._slice_dict_merge(self._legends_kw, legends_kw_extend)
        #else:
        legends_kw = self._legends_kw

        for ax_key, kws in legends_kw.items():
            if not kws is None:
                self._map_key(ax_key, lambda ax: ax.legend(**kws))

        if fig_legend_kw is None:
            if not self._fig_legend_kw is None:
                self.fig.legend(**self._fig_legend_kw)
        else:
            self.fig.legend(**(self._fig_legend_kw | fig_legend_kw))
            
        self.fig.savefig(path, **savefig_args)

    def show_nodes(self, path, figsize=(100, 60)):
        """
        Generate a graph containing the parent tree of
        all nodes that have been attached to this figure
        and save it into path. Optionally tune the figsize.
        """

        nodes_as_dict = {}
        for i, (node, (key, state, memo)) in enumerate(self._chains.items()):
            s = "{} on {}, state {}, len(memo)={}".format(i, key, state, len(memo))
            nodes_as_dict[s] = node

        group = NodeGroup("NodeFigure", parents=nodes_as_dict)

        draw_node_chain(group, path, figsize=figsize)

    def _pad_any(self, padding_factor, scale, lim):
        if scale == 'linear':
            inc = (lim[1] - lim[0]) * padding_factor / 2
            return lim[0] - inc, lim[1] + inc
        elif scale == 'log':
            return lim[0] * (1 - padding_factor), lim[1] * (1 + padding_factor)
        else:
            raise TypeError("Only lin or log axes support this padding algorithm")

    def _pad_oneside(self, padding_factor, scale, lim, side):
        if scale == 'linear':
            inc = (lim[1] - lim[0]) * padding_factor / 2
            if side == 'right':
                return lim[0], lim[1] + inc
            elif side == 'left':
                return lim[0] - inc, lim[1]
            else:
                raise ValueError("side must be either right or left")
        elif scale == 'log':
            if side == 'right':
                return lim[0], lim[1] * (1 + padding_factor)
            elif side == 'left':
                return lim[0] * (1 - padding_factor), lim[1] 
            else:
                raise ValueError("side must be either right or left")
        else:
            raise TypeError("Only lin or log axes support this padding algorithm")

    def _pad_right(self, padding_factor, ax):
        lim = ax.get_xlim()
        scale = ax.get_xscale()
        ax.set_xlim(self._pad_oneside(padding_factor, scale, lim, 'right'))

    def _pad_left(self, padding_factor, ax):
        lim = ax.get_xlim()
        scale = ax.get_xscale()
        ax.set_xlim(self._pad_oneside(padding_factor, scale, lim, 'left'))

    def _pad_top(self, padding_factor, ax):
        lim = ax.get_ylim()
        scale = ax.get_yscale()
        ax.set_ylim(self._pad_oneside(padding_factor, scale, lim, 'right'))

    def _pad_bottom(self, padding_factor, ax):
        lim = ax.get_ylim()
        scale = ax.get_yscale()
        ax.set_ylim(self._pad_oneside(padding_factor, scale, lim, 'left'))

    def _pad_x(self, padding_factor, ax):
        lim = ax.get_xlim()
        scale = ax.get_xscale()
        ax.set_xlim(self._pad_any(padding_factor, scale, lim))

    def _pad_y(self, padding_factor, ax):
        lim = ax.get_ylim()
        scale = ax.get_yscale()
        ax.set_ylim(self._pad_any(padding_factor, scale, lim))

    def pad(self, padding_factor, key=None, which='x'):
        if key is None:
            key = slice(None, None, None)

        ax_or_grid = self[key]

        try:
            ax_iter = iter(ax_or_grid)
        except TypeError:
            ax_iter = [ax_or_grid]

        for ax in ax_iter:
            if which == 'both' or which == 'all':
                self._pad_x(padding_factor, ax)
                self._pad_y(padding_factor, ax)
            else:
                if 'x' in which:
                    self._pad_x(padding_factor, ax)
                if 'y' in which:
                    self._pad_y(padding_factor, ax)
                if 't' in which:
                    self._pad_top(padding_factor, ax)
                if 'b' in which:
                    self._pad_bottom(padding_factor, ax)
                if 'r' in which:
                    self._pad_right(padding_factor, ax)
                if 'l' in which:
                    self._pad_left(padding_factor, ax)

    def _create(self, default_format):
        self._fig, self._axs = pplt.subplots(**({'share': False,'tight': True} | default_format.subplots))
        #{'sharex': False, 'sharey': False, 

        # if sharex and sharey is not set, proplot raises an error
        # tight layout is a good default
        self._fig.format(**(default_format.fig_format))
        for ax_key, ax_format in default_format.axs_format.items():
            self._axs[ax_key].format(**ax_format)

    def _map_key(self, key, callback):
        targets = self.axs[key]
        if isinstance(targets, pplt.SubplotGrid):
            for ax in targets: 
                rval = callback(ax)
        elif isinstance(targets, pplt.axes.Axes):
            rval = callback(targets)

        return rval

    def _execute_one_chain(self, node_chain, key, **kwargs):
        node_bound = lambda ax : node_chain(ax, **kwargs)
        return self._map_key(key, node_bound) 

