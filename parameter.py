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

def kwargs_dict(**kwargs):
        return kwargs

class ParameterStudyCreator:
    """
    Create a parameter study from a node chain.
    Copies the node chain given for each set of parameters that 
    are requested and returns different sets of parameters in
    different views.
    """
    def __init__(self, node_chain, default_params, params, times):
        self.node_chain = node_chain
        self.default_params = default_params
        self.params = params 
        self.times = times
        self.chains = {}
        self.groups = {}

    def time_evolutions(self, do_default=True):
        sep_runs = {}

        # generate timerange with default params
        if do_default:
            sep_runs['def'] = self.time_evolution_def()

        # generate names for not-default params (one changed at a time)
        sep_runs |= self.time_evolution()

        return sep_runs

    def _dynamic_filter(self, quantity, value):
        if not quantity is None:
            try:
                iter(quantity)
            except TypeError:
                quantity = [quantity]

            params = {k:v for k, v in self.params.items() if k in quantity}
        else:
            params = self.params

        if not value is None:
            try:
                iter(value)
            except TypeError:
                value = [value]

            for k in params:
                params[k] = [v for v in params[k] if v in value]

        return params

    def _get_chain(self, name, param):
        """
        Get the requested chain from self.chains if already copied,
        if not make a copy, store in self.chains and return it.
        """
        if not name in self.chains:
            self.chains[name] = self.node_chain.copy("_" + name, param=param)

        return self.chains[name]

    def _get_group(self, name, new_runs):
        if not name in self.groups:
            self.groups[name] = NodeGroup("group_{}".format(name), parents=new_runs)

        return self.groups[name]


    def time_evolution_def(self):
        new_runs = {}
        for t in self.times:
            name = "T={}_def".format(t)
            new_params = self.default_params | { 'Tmax' : t }
            new_runs[name] = self._get_chain(name, new_params)

        return {'def' : self._get_group("group_def", new_runs)}

    def time_evolution(self, quantity=None, value=None):
        params = self._dynamic_filter(quantity, value)

        sep_runs = {}
        for param, param_v in params.items():
            for v in param_v:
                new_runs = {}
                for t in self.times:
                    name = "T={}_{}={}".format(t, param, str(v))
                    new_params = self.default_params | { param : v, 'Tmax' : t }
                    new_runs[name] = self._get_chain(name, new_params)

                sep_runs["{}={}".format(param, v)] = self._get_group("group_{}={}".format(param, v), new_runs)

        return sep_runs

    def parameter_series(self, Tmax, quantity=None):
        params = self._dynamic_filter(quantity, None)
        sep_runs = {}

        if not Tmax in self.times:
            raise ValueError("Tmax wasn't generated (not found in times)")

        # generate names for not-default params (one changed at a time)
        for param, param_v in params.items():
            new_runs = {}
            for v in param_v + [self.default_params[param]]:
                if v == self.default_params[param]:
                    name = "T={}_def".format(Tmax)
                else:
                    name = "T={}_{}={}".format(Tmax, param, str(v))
                new_params = self.default_params | { param : v, 'Tmax' : Tmax }
                new_runs[name] = self._get_chain(name, new_params)

            sep_runs[param] = self._get_group("group_" + param, new_runs)

        return sep_runs
        
    def all_flat(self, do_default=True):
        runs = {}

        # generate timerange with default params
        if do_default:
            for t in self.times:
                name = "T={}_def".format(t)
                new_params = self.default_params | { 'Tmax' : t }
                runs[name] = self._get_chain(name, new_params)

        # generate names for not-default params (one changed at a time)
        for param, param_v in self.params.items():
            for v in param_v:
                for t in self.times:
                    name = "T={}_{}={}".format(t, param, str(v))
                    new_params = self.default_params | { param : v, 'Tmax' : t }
                    runs[name] = self._get_chain(name, new_params)

        return runs

class ParameterStudy:
    def __init__(self, node_chain, default_params, params, times):
        self.parameters = ParameterStudyCreator(node_chain, default_params, params, times)

    def plot_param_series(self, ax, quantity=None):
        groups = self.parameters.parameter_series(max(self.parameters.times), quantity)
        for t, group in groups.items():
            ax.format(title='Parameter Study: Parameter series for {}'.format(t))
            group(ax)

    def plot_time_evolution(self, ax, quantity=None, value=None):
        groups = self.parameters.time_evolution(quantity=quantity, value=value)
        for t, group in groups.items():
            ax.format(title='Parameter Study: Time evolution for {}'.format(t))
            group(ax)

    def plot_time_evolution_def(self, ax):
        group = self.parameters.time_evolution_def()
        ax.format(title='Parameter Study: Time evolution for default')
        group['def'](ax)

    
default_params = {
          'Xsh' : 0.002,
          'beta_s' : 0.08,
          'r' : 4,
          'dt' : 0.001,
          't_inj' : 0.00022, # -> sollten ca, 8 std sein
          'k_syn' : 0,
          'x0' : 0,
          'y0' : 1,
          'q' : 2
        }

params = {
    'beta_s' : [0.01, 0.06, 0.1, 0.6],
    'q' : [0.2, 1.5, 3, 20],
    'Xsh' : [0.0001, 0.001, 0.005, 0.05],
    't_inj' : [0.00042, 0.0017], # *2 /2
    'dt' : [0.0005, 0.0008, 0.0012, 0.002],
    'r' : [1.2, 2, 3.5, 5.5]
}

times = [0.64, 2.0, 6.4, 20, 200]

cyclex = ColorCycle(['red', 'green', 'blue', 'orange', 'black', 'violet'])
cyclep = ColorCycle(['red', 'green', 'blue', 'orange', 'black', 'violet'])
cache = PickleNodeCache('pickle', '9aparam')

batch = BatchNode('batch', batch_cls = PyBatchKruells9, cache=cache, ignore_cache=False)
points = PointNode('points', {'batch': batch}, cache=cache, ignore_cache=False)

valuesx = ValuesNode('valuesx', {'points' : points}, index=0, cache=cache, ignore_cache=False)
valuesp = ValuesNode('valuesp', {'points' : points}, index=1, cache=cache, ignore_cache=False,
                confinements=[(0, lambda x : np.abs(x) < 1)],
        )

histo_opts = kwargs_dict(bin_count=50, plot=True, cache=cache, ignore_cache=False)

histogramx = HistogramNode('histox',
                {'values' : valuesx}, 
                normalize='width', 
                color_cycle=cyclex,
                log_bins=False, 
                plot_kwargs={'linewidth' : 0.8},
                **histo_opts
        )
histogramp = HistogramNode('histop',
                {'values' : valuesp}, 
                normalize='density', 
                log_bins=True, 
                plot_kwargs={'alpha' : 0.6, 'linewidth' : 0.8},
                color_cycle=cyclep,
                **histo_opts
        )

powerlaw = PowerlawNode(
                'pl', 
                {'dataset' : histogramp},
                ln_x=False,
                plot=True,
                color_cycle=cyclep)

plotgroup = NodePlotGroup('plotgroup', parents=[histogramx, powerlaw])

paramstudy = ParameterStudy(plotgroup, default_params, params, times)

def do_one_plot(plot_dir, name, plot_cb):
    fig, axs = pplt.subplots(ncols=2, share=False, tight=True)
    axs.format(title=name, yscale='log',
               yformatter=pplt.SciFormatter())
    axs[1].format(xscale='log', xformatter=pplt.SciFormatter())
    plot_cb(axs)
    axs[1].legend(ncols=2, loc='r')
    fig.savefig("{}/{}.pdf".format(plot_dir, name))

plot_dir = "figures/9aparam"
do_one_plot(plot_dir, 'Time evolution', paramstudy.plot_time_evolution_def)
for param in paramstudy.parameters.params:
    name = 'param series {}'.format(param)
    logging.info("Plotting {}".format(name))
    do_one_plot(plot_dir, name,
            lambda axs : paramstudy.plot_param_series(axs, quantity=param))

for param, param_v in paramstudy.parameters.params.items():
    for v in param_v:
        name = 'Time evolution for {}={}'.format(param, v)
        logging.info("Plotting {}".format(name))
        do_one_plot(plot_dir, name,
                lambda axs : paramstudy.plot_time_evolution(axs, quantity=param, value=v))
    
