import proplot as pplt

pplt.rc.update({
                'text.usetex' : True,
                })


class NodeFigureFormat:
    def __init__(self, subplots=None, fig_format=None, axs_format=None, ax_format=None, legend_kw=None, legends_kw=None):
        """
        A set of format options for repeated use in NodeFigure instances.
        
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
        """
        if subplots is None:
            self.subplots = {}
        else:
            self.subplots = subplots

        if fig_format is None:
            self.fig_format = {}
        else:
            self.fig_format = fig_format

        self.axs_format = NodeFigureFormat._multi_param_parse(axs_format, ax_format)

        self.legends_kw = NodeFigureFormat._multi_param_parse(legends_kw, legend_kw)

    def __getitem__(self, key):
        return self.axs_format[key]

    def __repr__(self):
        return "NodeFigureFormat({}, {}, axs_format={}, legends_kw={})".format(self.subplots, self.fig_format, self.axs_format, self.legends_kw)

    def __str__(self):
        return repr(self)

    @staticmethod
    def _multi_param_parse(specific, dict_all):
        if specific is None:
            if dict_all is None:
                return {}
            else:
                return {slice(None, None, None) : dict_all}
        elif isinstance(specific, list):
            return dict(enumerate(specific))
        elif isinstance(specific, dict):
            return specific

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

        # _must_run contains the node_chains to be executed as keys and
        # the respective key for SubplotGrid.__getitem() to be executed on as values
        self._remaining_chains = {}

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
        Execute all scheduled node_chains now.
        """
        while len(self._remaining_chains) > 0:
            chain, key = self._remaining_chains.popitem()
            self._execute_one_chain(chain, key)

    def add(self, node_chain, to=None, instant=False):
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
        """
        if to is None:
            to = slice(None, None, None)

        if instant:
            self._execute_one_chain(node_chain, to)
        else:
            self._remaining_chains[node_chain] = to

    def savefig(self, path, legend_kw=None, legends_kw=None, savefig_args=None):
        """
        Save figure to path (using savefig_args as kwargs for savefig)
        after adding a legend where applicable.

        Node chains that have been added but not run are executed now. 

        legend_kw and legends_kw work exactly like NodeFigureFormat's equally
        named parameters. If one is set for this method, NodeFigureFormat's 
        parameters are both overridden. See in NodeFigureFormat for more
        information.

        """
        if savefig_args is None:
            savefig_args = {}

        self.execute()

        if not legend_kw is None or not legends_kw is None:
            legends_kw = NodeFigureFormat._multi_param_parse(legends_kw, legend_kw)
        else:
            legends_kw = self._legends_kw

        for ax_key, kws in legends_kw.items():
            self._map_key(ax_key, lambda ax: ax.legend(**kws))
            

        self.fig.savefig(path, **savefig_args)

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
                callback(ax)
        elif isinstance(targets, pplt.axes.Axes):
            callback(targets)


    def _execute_one_chain(self, node_chain, key):
        self._map_key(key, node_chain) 

