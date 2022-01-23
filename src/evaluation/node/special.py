import sys
import numbers
from .node import EvalNode, dict_or_list_iter
sys.path.insert(0, '../../../lib')
sys.path.insert(0, '../../../../lib')
sys.path.insert(0, '../../lib')
sys.path.insert(0, 'lib')
sys.path.insert(0, '../lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.pybreakpointstate import *
from evaluation.helpers import *
from evaluation.experiment import *
from scipy import stats

def kieslinregress(x, y, y_err=None):
    if y_err is None:
        # no y uncertainties given
        # number of samples
        N = len(x)
        determinant = N * np.sum(x**2) - np.sum(x)**2
        # intercept
        a = 1/determinant * ( np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x*y) )
        # slope
        b = 1/determinant * ( N * np.sum(x*y) - np.sum(x) * np.sum(y) )
        # standard error of the regression
        s = np.sqrt( 1/(N - 2) * np.sum( (y - a - b*x)**2 ) )
        # uncertainty of intercept
        a_err = 1/determinant * s**2 * np.sum(x**2)
        # uncertainty of slope
        b_err = 1/determinant * s**2 * N
        # correlation coefficient
        # from https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample
        r = ( np.sum(x*y) - N * np.mean(x) * np.mean(y) ) / np.sqrt( ( np.sum(x**2) - N * np.mean(x)**2 ) * ( np.sum(y**2) - N * np.mean(y)**2 ) )
        # slope, intercept, slope_err, intercept_err, s, r = kiesregress(x, y)
        return b, a, b_err, a_err, s, r
 
 
    # y uncertainties given
    # number of samples
    N = len(x)
    determinant = np.sum( 1 / (y_err**2) ) * np.sum( (x / y_err)**2 ) - np.sum( x / (y_err**2) )**2
    # intercept
    a = 1/determinant * ( np.sum( y / (y_err**2) ) * np.sum( (x / y_err)**2 ) - np.sum( x / (y_err**2) ) * np.sum( x * y / (y_err**2) ) )
    # slope
    b = 1/determinant * ( np.sum( 1 / (y_err**2) ) * np.sum( x * y / (y_err**2) ) - np.sum( x / (y_err**2) ) * np.sum( y / (y_err**2) ) )
    # uncertainty of intercept
    a_err = np.sqrt( 1/determinant * np.sum( (x / y_err)**2 ) )
    # uncertainty of slope
    b_err = np.sqrt( 1/determinant * np.sum( 1 / y_err**2 ) )
    # standard error of the regression
    s = np.sqrt( 1/(N - 2) * np.sum( (y - a - b*x)**2 ) )
    # correlation coefficient
    # from https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample
    r = ( np.sum(x*y) - N * np.mean(x) * np.mean(y) ) / np.sqrt( ( np.sum(x**2) - N * np.mean(x)**2 ) * ( np.sum(y**2) - N * np.mean(y)**2 ) )
    # slope, intercept, slope_err, intercept_err, s, r = kiesregress(x, y, y_err)
    return b, a, b_err, a_err, s, r

class BatchNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        exset = Experiment(
                    kwargs['batch_cls'], 
                    kwargs['param'],
                )
        exset.run()

        return exset
    def def_kwargs(self, **kwargs):
        kwargs = {
            'label_fmt_fields' : {}
        } | kwargs

        return kwargs

    def common(self, common, **kwargs):
        return {'batch_param' : kwargs['param'], 'label_fmt_fields' : kwargs['label_fmt_fields']}

class PointNode(EvalNode):
    """
    Retrieves all points of particles from a batch
    """
    def do(self, parent_data, common, end_state=PyBreakpointState.TIME):
        experiment = parent_data['batch']
        return [p.x for p in experiment.states if p.breakpoint_state == end_state]

class ValuesNode(EvalNode):
    """
    Reqiured parents:
        experiment

    Returns:
        list of all values
    """
    def do(self, parent_data, common, index, confinements=[], end_state=PyBreakpointState.TIME):
        points = parent_data['points'] 
        for conf_idx, conf_cond in confinements:
            points = [p for p in points if conf_cond(p[conf_idx])]
            
        return [p[index] for p in points]


#class ConfineNode(EvalNode):
#    def do(self, parent_data, common, confinements):
#        points = parent_data['points']
#
#        for conf_idx, conf_cond in confinements:
#            points = [p for p in relevant_pps if conf_cond(p.x[conf_idx])]

        

#class TimelimitWeightsEvalNode(EvalNode):
#    def do(self, parent_data, index, use_integrator=None, confinements=[]):
#        experiment = parent_data['experiment'] 
#
#        if use_integrator is None:
#            weights = np.array([1] * len(experiment.states))
#        else:
#            weights = experiment.integrator_values[use_integrator]
#
#        relevant_states_weights = zip(experiment.states, weights)
#        for conf_idx, conf_cond in confinements:
#            relevant_states_weights = [(p, w) for p, w in zip(experiment.states, weights) if conf_cond(p.x[conf_idx])]
#
#        relevant_weights = [w for p, w in relevant_states_weights if p.breakpoint_state == self.options['end_state']]
#        return relevant_weights

class HistogramNode(EvalNode):
    @staticmethod
    def _get_bin_count(bin_count, average_bin_size, n_states):
        if not bin_count is None:
            return int(bin_count)
        else:
            c = int(n_states / average_bin_size)
            return c if c != 0 else 1

    def def_kwargs(self, **kwargs):
        kwargs = {
            'bin_count' : None,
            'average_bin_size' : 100,
            'normalize' : 'density',
            'auto_normalize' : None,
            'manual_normalization_factor' : 1,
            'transform' : None,
            'log_bins' : False,
            'label' : 'Tmax={Tmax}',
            'style' : 'step',
            'show_errors' : True,
            'plot_kwargs' : {},
        } | kwargs

        if not kwargs['auto_normalize'] is None:
            logging.warning("The use of auto_normalize is deprecated. The parameter is ignored. Use normalize='density' instead.")

        return kwargs

    def common(self, common, **kwargs):
        if 'color_cycle' in kwargs:
            color = next(kwargs['color_cycle'])
        else:
            color = None

        return {'color' : color}

    def do(self, parent_data, common, **kwargs):
        experiment = parent_data['values']

        rev = parent_data['values']
        if 'weights' in parent_data:
            weights = parent_data['weights']
        else:
            weights = np.ones(len(rev))

        bin_count = type(self)._get_bin_count(kwargs['bin_count'], kwargs['average_bin_size'], len(rev))
        rev_dim = np.array(rev).T[0]

        if kwargs['log_bins']:
            bins = np.logspace(np.log10(min(rev_dim)), np.log10(max(rev_dim)), bin_count + 1)
        else:
            bins = np.linspace(min(rev_dim), max(rev_dim), bin_count + 1)

        try:
            histogram, edges = np.histogram(rev_dim, bins=bins, weights=weights, density=False)
        except ValueError as e:
            print("verr", len(weights), len(rev_dim), "NaN: ", np.count_nonzero(np.isnan(arr)), np.isnan(arr), rev_dim)
            raise e
        
        if kwargs['normalize'] == 'density':
            db = np.array(np.diff(edges), float)
            norm = 1 / db / histogram.sum()
        elif kwargs['normalize'] == 'width':
            db = np.array(np.diff(edges), float)
            norm = 1 / db
        elif isinstance(kwargs['normalize'], numbers.Number):
            norm = kwargs['normalize']
        else:
            norm = 1

        errors = np.sqrt(histogram) * norm * kwargs['manual_normalization_factor']
        histogram = histogram * norm * kwargs['manual_normalization_factor']

        param = edges[:-1] + np.diff(edges) / 2

        if not kwargs['transform'] is None:
            param, histogram = kwargs['transform'](param, histogram)

        return param, histogram, errors

    def plot(self, v, ax, common, **kwargs):
        param, histogram, errors = v

        fmt_fields = common['batch_param'] | common['label_fmt_fields']
        label = kwargs['label'].format(**(fmt_fields))

        if kwargs['style'] == 'step':
            ax.step(param, histogram, color=common['color'], label=label, **(kwargs['plot_kwargs']))
        elif kwargs['style'] == 'line':
            shadedata = errors if kwargs['show_errors'] else None
            ax.plot(param, histogram, color=common['color'], shadedata=shadedata, label=label, **(kwargs['plot_kwargs']))
        else:
            raise ValueError("Invalid histogram plot style used. Valid: step, line")

class PowerlawNode(EvalNode):
    def def_kwargs(self, **kwargs):
        kwargs = {
            'ln_x' : False,
            'label' : "",
            'powerlaw_annotate' : True,
            'errors' : False
        } | kwargs

        return kwargs

    def do(self, parent_data, common, **kwargs):
        if kwargs['errors'] or len(parent_data['dataset']) == 3:
            x, y, errors = parent_data['dataset']
        else:
            x, y = parent_data['dataset']
        

        # cut data at the first empty bin
        # most physical solution imho, because ignoring zeroes
        # is not really good
        if np.count_nonzero(y) == len(y):
            max_index = len(y)
        else:
            max_index = np.argmax(y == 0)

        x = x[:max_index]
        y = y[:max_index]
        if kwargs['errors']:
            errors = errors[:max_index]

        lims = ((min(x), max(x)), (min(y), max(y)))

        if not kwargs['ln_x']:
            x = np.log(x)

        y = np.log(y)
        
        if kwargs['errors']:
            m, t, dm, dt, _, _ = kieslinregress(x, y, errors)
        else:
            result = stats.linregress(x, y)
            m = result.slope
            t = result.intercept
            dm = result.stderr
            dt = result.intercept_stderr

        return np.exp(t), m, np.exp(t) * dt, dm, lims

    def plot(self, data, ax, common, **kwargs):
        a, q, _, dq, lims = data
        if kwargs['ln_x']:
            func = lambda x : a * np.exp(x*q)
        else:
            func = lambda x : a * x**q


        label = kwargs['label']
        if kwargs['powerlaw_annotate']:
            label += ' $q={:.2f}\\pm{:.2f}$'.format(q, dq)

        if 'color' in common:
            color = common['color']
        else:
            color = None

        x_plot = np.linspace(lims[0][0], lims[0][1], 100)
        y_plot = func(x_plot)
        x_plot = [x for x, y in zip(x_plot, y_plot) if y <= lims[1][1] and y >= lims[1][0]]
        y_plot = [y for y in y_plot if y <= lims[1][1] and y >= lims[1][0]]
        ax.plot(x_plot, y_plot, label=label, color=color, linestyle='dotted')


class CommonParamNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        return common['param'][kwargs['key']]


class ScatterNode(EvalNode):
    """
    This node collects data from a list of parents and
    presents them in a scatter plot with optional y errors.

    Parent specification:
    Expects an iterable of parents in which each item
    can be subscripted with 'x' and 'y' and optional 'dy'
    for y errors.

    kwargs:
     * label: The label of the datarow

    """
    def def_kwargs(self, **kwargs):
        kwargs = {
            'label' : '',
        } | kwargs

        return kwargs

    def do(self, parent_data, common, **kwargs):
        x = []
        y = []
        yerr = []
        for _, datapoint in dict_or_list_iter(parent_data):
            if type(datapoint) == list:
                x_, y_ = datapoint
                yerr_ = None
            else:
                x_ = datapoint['x']
                y_ = datapoint['y']

                if 'dy' in datapoint:
                    yerr_ = datapoint['dy']
                else:
                    yerr_ = None

            x.append(x_)
            y.append(y_)
            yerr.append(yerr_)

        return x, y, yerr

    def plot(self, data, ax, common, **kwargs):
        x, y, dy = data

        if np.all(dy is None):
            bardata = None
        elif np.any(dy is None):
            bardata = [0 if b is None else b for b in bardata]
        else:
            bardata = dy

        ax.plot(x, y, bardata=bardata, label=kwargs['label'], lw=1, barlw=0.5, marker='x', capsize=1.0)

class HistogramIntegrateNode(HistogramNode):
    def def_kwargs(self, **kwargs):
        kwargs = {
            'label' : 'Tmax={Tmax}',
            'style' : 'step',
            'show_errors' : None,
            'plot_kwargs' : {},
            'from_inf' : True
        } | kwargs

        if not kwargs['show_errors'] is None:
            logging.warning("Errors in integrated histograms are not available at the moment.")

        return kwargs

    def do(self, parent_data, common, **kwargs):
        y_int = []
        y_track = 0
        x = parent_data['histogram'][0]
        y = parent_data['histogram'][1]
        from_inf = kwargs['from_inf']

        for i, x_ in enumerate(x):
            y_track = 0
            if from_inf:
                elem = [y__ for x__, y__ in zip(x, y) if x__ >= x_]
            else:
                elem = [y__ for x__, y__ in zip(x, y) if x__ <= x_]
            y_int.append(sum(elem))

        return x, y_int, [0] * len(y)

