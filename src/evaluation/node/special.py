import sys
import numbers
import copy
from .node import EvalNode, dict_or_list_iter, dict_or_list_map
sys.path.insert(0, '../../../lib')
sys.path.insert(0, '../../../../lib')
sys.path.insert(0, '../../lib')
sys.path.insert(0, 'lib')
sys.path.insert(0, '../lib')
sys.path.insert(0, 'src/evaluation')
from pybatch.pybreakpointstate import *
from evaluation.helpers import *
from evaluation.experiment import *
from .cache import FileCache
from scipy import stats, optimize
from astropy import units as u
from astropy import constants
from agnpy.synchrotron.synchrotron import Synchrotron
from agnpy.compton.synchrotron_self_compton import SynchrotronSelfCompton
from agnpy.spectra import PowerLaw
from functools import cache

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

class PhysBatchNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        pyb = kwargs['batch_cls'](phys_params=kwargs['phys_params'], num_params=kwargs['num_params'])
        exset = Experiment(pyb, pyb.batch_params)
        exset.run()

        return exset

    def def_kwargs(self, **kwargs):
        kwargs = {
            'label_fmt_fields' : {},
        } | kwargs

        #if not 'phys_params' in kwargs or not 'num_params' in kwargs:
        #    raise ValueError("phys_params and num_params must be passed")

        return kwargs

    def common(self, common, **kwargs):
        phy = kwargs['phys_params']
        num = kwargs['num_params']
        par = kwargs['batch_cls'].get_batch_params(phy, num)

        return {'batch_param' : par, 'phys_param' : phy, 'num_param' : num, 'label_fmt_fields' : kwargs['label_fmt_fields'] | par | phy | num }

class BatchNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        exset = Experiment(
                    kwargs['batch_cls'], 
                    kwargs['param'],
                )
        exset.run(kwargs['nthreads'])

        return exset

    def def_kwargs(self, **kwargs):
        kwargs = {
            'label_fmt_fields' : {},
            'nthreads' : 1
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
        a = np.array([p.x for p in experiment.states if p.breakpoint_state == end_state])
        aback = [e for e in a if e[1] <= 1]
        if len(aback) > 0:
            #print("backpropagation detected in {}/{} particles at node {}. printing values: ".format(len(aback), len(a), self.name), aback)
            print("backpropagation detected in {}/{} particles at node {}".format(len(aback), len(a), self.name))
        return a


class ValuesNode(EvalNode):
    """
    Reqiured parents:
        points

    Lambdas in confinements are not pickle-able so there
    is confine_range which takes a list of
        (index, min_value, max_value)
    tuples if you want to use kwargs pickling

    Returns:
        list of all values
    """
    #def def_kwargs(self, **kwargs):
    #    kwargs = {
    #            'confinements': [],
    #            'end_state': PyBreakpointState.TIME
    #        } | kwargs
     
    #    return kwargs

    def do(self, parent_data, common, index, confinements=[], confine_range=[], end_state=PyBreakpointState.TIME, **kwargs):
        points = parent_data['points'] 
        for conf_idx, conf_cond in confinements:
            points = [p for p in points if conf_cond(p[conf_idx])]

        for conf_idx, conf_min, conf_max in confine_range:
            points = [p for p in points if conf_min <= p[conf_idx] and conf_max >= p[conf_idx]]

        v_array = np.array([p[index] for p in points])

        if index == 1:
            # numerical error detection
            a = np.count_nonzero(v_array <= 1)
            b = np.count_nonzero(v_array <= 0)
            if a > 0:
                print("decreasing momentum detected in {}/{} particles at node {}".format(a, len(v_array), self.name))
            if b > 0:
                print("negative momentum detected in {}/{} particles at node {}".format(b, len(v_array), self.name))

        return v_array


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

class FunctionNode(EvalNode):
    def def_kwargs(self, **kwargs):
        if not 'callback' in kwargs:
            raise ValueError("FunctionNode needs a callback as kwarg")

        kwargs = {
            'detail' : 100,
            'label' : 'comparison'
        } | kwargs

        return kwargs

    def do(self, parent_data, common, **kwargs):
        return None

    def plot(self, v, ax, common, **kwargs):
        xvals = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], kwargs['detail'])
        yvals = kwargs['callback'](xvals)
        line = ax.plot(xvals, yvals, label=kwargs['label'])[0]
        return line

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
            'bin_width' : None,
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

    #def common(self, common, **kwargs):
    #    if not 'color' in common:
    #        return {'color' : None}

        #if 'color_cycle' in kwargs:
        #    color = next(kwargs['color_cycle'])
        #else:
        #    color = None

        #return {'color' : color}

    def do(self, parent_data, common, **kwargs):
        rev = parent_data['values']

        if 'weights' in parent_data:
            weights = parent_data['weights']
        else:
            weights = np.ones(len(rev))

        rev_dim = np.array(rev).T[0]
        if kwargs['bin_width'] is None:
            bin_count = type(self)._get_bin_count(kwargs['bin_count'], kwargs['average_bin_size'], len(rev))
            if kwargs['log_bins']:
                #print(min(rev_dim), max(rev_dim))
                bins = np.logspace(np.log10(min(rev_dim)), np.log10(max(rev_dim)), bin_count + 1)
            else:
                bins = np.linspace(min(rev_dim), max(rev_dim), bin_count + 1)
        else:
            if kwargs['log_bins']:
                bins = 10**np.arange(np.log10(min(rev_dim)), np.log10(max(rev_dim)), kwargs['bin_width'])
            else:
                bins = np.arange(min(rev_dim), max(rev_dim), kwargs['bin_width'])

        try:
            histogram, edges = np.histogram(rev_dim, bins=bins, weights=weights, density=False)
        except ValueError as e:
            print("verr", len(weights), len(rev_dim), "NaN: ", np.count_nonzero(np.isnan(arr)), np.isnan(arr), rev_dim)
            raise e
        #print(self.name, rev, bins, histogram)
        
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
        #print("hist2", errors/histogram)

        param = edges[:-1] + np.diff(edges) / 2

        if not kwargs['transform'] is None:
            param, histogram = kwargs['transform'](param, histogram)

        return param, histogram, errors, edges

    def plot(self, v, ax, common, **kwargs):
        param, histogram, errors, edges = v

        if ValuesNode in common['_kwargs_by_type']:
            add_fields = common['_kwargs_by_type'][ValuesNode]
        else:
            add_fields = {}

        fmt_fields = {}

        if 'batch_param' in common:
            fmt_fields |= common['batch_param']
        if 'label_fmt_fields' in common:
            fmt_fields |= common['label_fmt_fields']

        fmt_fields |= add_fields

        label = kwargs['label'].format(**(fmt_fields))

        #print(common['label_fmt_fields'])
        if kwargs['style'] == 'step':
            if kwargs['show_errors']:
                hupper = histogram + errors
                hlower = histogram - errors
                ax.fill_between(param, hlower, hupper, step='mid', color=self.get_color(), alpha=0.5)
            # proplot where parameter not working (see github issue)
            # using edges[1:] yields the same behaviour where='mid' should
            lines = ax.step(edges[1:], histogram, label=label, color=self.get_color(), linewidth=0.7, **kwargs['plot_kwargs'])
        elif kwargs['style'] == 'line':
            shadedata = errors if kwargs['show_errors'] else None
            lines = ax.plot(param, histogram, shadedata=shadedata, label=label, color=self.get_color(), **kwargs['plot_kwargs'])
        else:
            raise ValueError("Invalid histogram plot style used. Valid: step, line")

        self.set_color(lines[0].get_color())

        return lines[0]

class Histogram2DNode(HistogramNode):
    @staticmethod
    def _get_bin_count(bin_count, average_bin_size, n_states):
        if not bin_count is None:
            return int(bin_count)
        else:
            return int(np.sqrt(HistogramNode._get_bin_count(bin_count, average_bin_size, n_states)))

    def def_kwargs(self, **kwargs):
        kwargs = {
            'bin_count' : (None, None),
            'log_bins' : (False, False),
            'log_histogram' : False,
            'cmap' : None,
            'limits' : ((-np.inf, np.inf), (-np.inf, np.inf))
        } | kwargs

        kwargs = super().def_kwargs(**kwargs)

        if not type(kwargs['bin_count']) is tuple:
            kwargs['bin_count'] = (kwargs['bin_count'], kwargs['bin_count'])

        if not type(kwargs['log_bins']) is tuple:
            kwargs['log_bins'] = (kwargs['log_bins'], kwargs['log_bins'])

        if not kwargs['auto_normalize'] is None:
            logging.warning("The use of auto_normalize is deprecated. The parameter is ignored. Use normalize='density' instead.")

        return kwargs

    def _prepare_onedim(self, rev, dim, **kwargs):
        bin_count = type(self)._get_bin_count(kwargs['bin_count'][dim], kwargs['average_bin_size'], len(rev))
        rev_dim = np.array(rev).T[0]

        if kwargs['log_bins'][dim]:
            bins = np.logspace(np.log10(min(rev_dim)), np.log10(max(rev_dim)), bin_count + 1)
        else:
            bins = np.linspace(min(rev_dim), max(rev_dim), bin_count + 1)

        return rev_dim, bins

    def filter_values(self, revx, revy, lims):
        xf = []
        yf = []
        for x_, y_ in zip(revx, revy):
            if lims[0][0] <= x_ and x_ <= lims[0][1] and lims[1][0] <= y_ and y_ <= lims[1][1]:
                xf.append(x_)
                yf.append(y_)

        return np.array(xf), np.array(yf)

    def do(self, parent_data, common, **kwargs):
        vx = parent_data['valuesx']
        vy = parent_data['valuesy']
        if not len(vx) == len(vy):
            raise ValueError("Histogram2D parents need same dimension")

        revx, revy = self.filter_values(vx, vy, kwargs['limits'])

        revxd, bins_x = self._prepare_onedim(revx, 0, **kwargs)
        revyd, bins_y = self._prepare_onedim(revy, 1, **kwargs)

        try:
            histogram, xedges, yedges = np.histogram2d(revxd, revyd, bins=[bins_x, bins_y], density=False)
        except ValueError as e:
            print("verr", len(revxd), revxd)
            raise e

        histogram = histogram.T # neccessery somehow, see docs of histogram2d
        
        if kwargs['normalize'] == 'density':
            dbx = np.array(np.diff(xedges), float)
            dby = np.array(np.diff(yedges), float)
            areas = dby.reshape((len(dby), 1)) * dbx # maybe the other way round? not sure -> this seems to be right comparing with numpy density=True
            norm = 1 / areas / histogram.sum()
        elif kwargs['normalize'] == 'width':
            dbx = np.array(np.diff(xedges), float)
            dby = np.array(np.diff(yedges), float)
            areas = dbx.reshape((len(dbx), 1)) * dby # maybe the other way round? not sure
            norm = 1 / areas
        elif isinstance(kwargs['normalize'], numbers.Number):
            norm = kwargs['normalize']
        else:
            norm = 1

        errors = np.sqrt(histogram) * norm * kwargs['manual_normalization_factor']
        histogram = histogram * norm * kwargs['manual_normalization_factor']

        if kwargs['log_histogram']:
            histogram = np.log10(histogram)

        paramx = xedges[:-1] + np.diff(xedges) / 2
        paramy = yedges[:-1] + np.diff(yedges) / 2
        #histogram = paramy**3 * histogram

        if not kwargs['transform'] is None:
            paramx, paramy, histogram = kwargs['transform'](paramx, paramy, histogram)

        return paramx, paramy, histogram, errors, xedges, yedges

    def plot(self, v, ax, common, **kwargs):
        paramx, paramy, histogram, errors, xedges, yedges = v

        if ValuesNode in common['_kwargs_by_type']:
            add_fields = common['_kwargs_by_type'][ValuesNode]
        else:
            add_fields = {}

        fmt_fields = common['batch_param'] | common['label_fmt_fields'] | add_fields
        label = kwargs['label'].format(**(fmt_fields))

        #print(common['label_fmt_fields'])
        if kwargs['style'] == 'contour':
            lines = ax.contour(paramx, paramy, histogram, label=label, cmap=kwargs['cmap'], **kwargs['plot_kwargs'])
        elif kwargs['style'] == 'contourf':
            lines = ax.contourf(paramx, paramy, histogram, label=label, cmap=kwargs['cmap'], **kwargs['plot_kwargs'])
        elif kwargs['style'] == 'heatmap':
            lines = ax.pcolormesh(xedges, yedges, histogram, label=label, cmap=kwargs['cmap'], **kwargs['plot_kwargs'])
        elif kwargs['style'] == 'surface':
            x, y = np.meshgrid(xedges, yedges)
            lines = ax.plot_surface(x, y, histogram, label=label, cmap=kwargs['cmap'], **kwargs['plot_kwargs'])
        else:
            raise ValueError("Invalid histogram2d plot style used. Valid: contour, heatmap, surface")

        #self.set_color(lines[0].get_color())

        return lines

class PowerlawNode(EvalNode):
    def def_kwargs(self, **kwargs):
        kwargs = {
            'ln_x' : False,
            'label' : "",
            'powerlaw_annotate' : True,
            'errors' : False,
            'error_type' : 'kiessling',
            'plot_kwargs': {},
            'negative' : False,
            'ndigits' : 3
        } | kwargs

        return kwargs

    def do(self, parent_data, common, **kwargs):
        if kwargs['errors'] or len(parent_data['dataset']) == 4:
            x, y, errors, _ = parent_data['dataset']
        else:
            x, y = parent_data['dataset']
            errors = None

        # cut data at the first empty bin
        # most physical solution imho, because ignoring zeroes
        # is not really good
        if np.count_nonzero(y) == len(y):
            max_index = len(y)
        else:
            max_index = np.argmax(y == 0)

        x = x[:max_index]
        y = y[:max_index]
        if not errors is None:
            errors = errors[:max_index]

        lims = ((min(x), max(x)), (min(y), max(y)))

        if not kwargs['ln_x']:
            x = np.log(x)

        if not errors is None:
            errors = errors / y # dln(x) = dx / x
        y = np.log(y)
        
        if not errors is None and kwargs['error_type'] == 'kiessling':
            m, t, dm, dt, _, _ = kieslinregress(x, y, errors)
        elif not errors is None and kwargs['error_type'] == 'scipy':
            popt, pcov = optimize.curve_fit(lambda x, m, t : m * x + t, x, y, sigma=errors)
            m, t = popt
            dm, dt = np.sqrt(np.diag(pcov))
        else:
            result = stats.linregress(x, y)
            m = result.slope
            t = result.intercept
            dm = result.stderr
            dt = result.intercept_stderr

        if kwargs['negative']:
            m *= -1

        return np.exp(t), m, np.exp(t) * dt, dm, lims

    def common(self, common, **kwargs):
        if not 'color' in common:
            return {'color' : None}

    def plot(self, data, ax, common, **kwargs):
        a, q, _, dq, lims = data

        if kwargs['negative']:
            q  *= -1

        if kwargs['ln_x']:
            x_plot = np.linspace(lims[0][0], lims[0][1], 100)
            y_plot = a * np.exp(q * x_plot)
        else:
            x_plot = np.logspace(np.log10(lims[0][0]), np.log10(lims[0][1]), 100)
            y_plot = a * x_plot**q


        label = kwargs['label']
        if kwargs['powerlaw_annotate']:
            label += ' $s={:.{}f}\\pm {:.{}f}$'.format(q, kwargs['ndigits'], dq, kwargs['ndigits'])

        x_plot = [x for x, y in zip(x_plot, y_plot) if y <= lims[1][1] and y >= lims[1][0]]
        y_plot = [y for y in y_plot if y <= lims[1][1] and y >= lims[1][0]]

        style = dict(linestyle='dotted', color=self.get_color()) | kwargs['plot_kwargs']

        line = ax.plot(x_plot, y_plot, label=label, **style) 

        self.set_color(line[0].get_color())

        return line[0]

class MapNode(EvalNode):
    def get_callback(self, **kwargs):
        if not 'callback' in kwargs:
            raise ValueError("No callback given")
        else:
            return kwargs['callback']

    def do(self, parent_data, common, **kwargs):
        cb = self.get_callback(**kwargs)
        return cb(parent_data)

class LimitNode(MapNode):
    def def_kwargs(self, **kwargs):
        kwargs = {
            'key_compare' : 0,
            'key_other' : 1,
            'upper' : np.inf,
            'lower' : -np.inf
        } | kwargs
        return kwargs

    def get_callback(self, **kwargs):
        def cb(parent_data):
            comp = kwargs['key_compare']
            idx = np.nonzero(np.logical_and(kwargs['lower'] <= parent_data[comp], parent_data[comp] <= kwargs['upper']))

            try:
                other = list(kwargs['key_other'])
            except TypeError:
                other = [kwargs['key_other']]

            new_parent_data = list(copy.copy(parent_data))
            for o in other:
                new_parent_data[o] = parent_data[o][idx]
            new_parent_data[comp] = parent_data[comp][idx]

            return new_parent_data

        return cb

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
            'plot_kwargs' : {},
            'value_limits' : (-np.inf, np.inf),
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

            if kwargs['value_limits'][0] <= y_ and y_ <= kwargs['value_limits'][1]:
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

        def_kw = dict(lw=1, barlw=0.5, marker='x', capsize=1.0)
        lines = ax.plot(x, y, bardata=bardata, label=kwargs['label'], **(def_kw | kwargs['plot_kwargs']))
        return lines[0]

class CutoffFindNode(EvalNode):
    def def_kwargs(self, **kwargs):
        kwargs = {
            'reverse' : False,
            'sigmas' : 1,
            'negative' : True,
        } | kwargs

        return kwargs

    def do(self, parent_data, common, **kwargs):
        x, y, _ = parent_data
        s = kwargs['sigmas']
        y = np.array(y)[np.argsort(x)]
        if kwargs['reverse']:
            x = np.flip(x)
            y = np.flip(y)
        
        count = 2
        while count < len(y) and np.abs(y[count] - np.mean(y[:count])) < s * np.std(y[:count]):
            count += 1

        m = (y[count] - y[count - 1]) / (x[count] - x[count - 1])
        if y[count] < np.mean(y[:count]) - s * np.std(y[:count]):
            y_cutoff = np.mean(y[:count]) - s * np.std(y[:count])
        else:
            y_cutoff = np.mean(y[:count]) + s * np.std(y[:count])

        x_cutoff = (y_cutoff - y[count - 1] + m * x[count - 1]) / m

        print(self.name, x_cutoff, y_cutoff)
        return x_cutoff, y_cutoff

    def plot(self, data, ax, common, **kwargs):
        x, y = data
        print('asdf')
        ax.axhline(y)
        ax.axvline(x)
        return ax.axhline(y), ax.axvline(x)



        


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

class PointsNodeCache(FileCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extension = ".values"

    @staticmethod
    def _read_file(file):
        pass

    @staticmethod
    def _write_file(file, obj):
        ps = np.array(obj).T[0].T
#        print(ps)

class PointsMergeNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        cc = np.concatenate(parent_data)
        lenstr = str([len(p) for p in parent_data])
        print("collecting points from {} runs having a total of {} particles. the runs have the following particle count: {}".format(len(parent_data), len(cc), lenstr))
        return cc

class MomentumCount(EvalNode):
    def do(self, parent_data, common, **kwargs):
        y, U, err, edges = parent_data
        return kwargs['p_inj'] * np.array(y), np.array(y)**1 * np.array(U) * u.Unit("cm-3"), np.array(err), np.array(edges) * kwargs['p_inj']

    def plot(self, data, ax, common, **kwargs):
        _, U, _, p = data
        return ax.plot(p[1:], U)[0]

class HistogramElectronDistribution:
    """
    Imitating the agnpy interface of an electron distribution
    """
    def __init__(self, p_edges, n_data):
        self.p_edges = p_edges
        self.n_data = n_data
        self.vfunc = np.vectorize(self._evaluate_one)

    @property
    def parameters(self):
        return []

    @cache
    @staticmethod
    def gamma_to_p(gamma):
        v = gamma * constants.m_e * constants.c # np.sqrt(gamma**2 - 1) * constants.m_e * constants.c
        return v

    def evaluate(self, gamma):
        if gamma.shape == ():
            return self._evaluate_one(gamma) * u.Unit("cm-3")
        else:
            vs = []
            for g in gamma.flatten():
                v =  self._evaluate_one(g)
                vs.append(v)
            ra = np.array(vs).reshape(gamma.shape)
            return ra * u.Unit("cm-3")

    @cache
    def _evaluate_one(self, gamma):
        p_ = HistogramElectronDistribution.gamma_to_p(gamma)
        for e0, e1, n in zip(self.p_edges[:-1], self.p_edges[1:], self.n_data):
            if e0 <= p_ and e1 > p_:
                return n / u.Unit("cm-3")

        return 0 

    def __call__(self, gamma):
        return self.evaluate(gamma)


        



class SynchrotronBase(EvalNode):
    #def get_nu_range(self, kwargs, parent_data):
    #    if not 'nu_range' in kwargs:
    #        return parent_data[0]

    def def_kwargs(self, **kwargs):
        kwargs = {
                'gamma_integrate' : np.logspace(1, 9, 200),
                'plot_kwargs' : {},
                'model_params_callback' : None,
                'factor' : 1,
                'label' : 'synchrotron'
            } | kwargs

        if not 'model_params' in kwargs:
            raise("Need model_params for calculating synchrotron radiation")

        return kwargs

    #def common(self, common, **kwargs):
    #    common['label_fmt_fields'].update({'B' : kwargs['model_params']['B']})

    def do(self, parent_data, common, **kwargs):
        if kwargs['model_params_callback'] is None:
            mp = kwargs['model_params']
        elif 'phys_param' in common:
            mp = kwargs['model_params_callback'](kwargs['model_params'], common['phys_param'])
        else:
            mp = kwargs['model_params_callback'](kwargs['model_params'], common['batch_param'])

        if self.parents_contains('N_data'):
            electron_dist = HistogramElectronDistribution(parent_data['N_data'][3], parent_data['N_data'][1])
        elif 'electron_dist' in kwargs:
            electron_dist = kwargs['electron_dist']
        else:
            raise IndexError('SynchrotronNode and derived nodes need a parent N_data supplying the histogram of electrons or a \'electron_dist\' kwarg with a compatible distribution object')

        logging.info("Calculating flux")
        nus = kwargs['nu_range']
        synchro = []

        #print("mp: ", mp)
        for nu in nus:
            flux = self.flux_from_nu(nu, electron_dist, mp, kwargs['gamma_integrate'])
            synchro.append(flux)
        #synchro = self.flux_from_nu(nus, parent_data['N_data'], mp, kwargs['gamma_integrate'])
        logging.info("Finished calculating flux")

        return u.Quantity(nus), u.Quantity(synchro) * kwargs['factor']

    def plot(self, data, ax, common, **kwargs):
        #print('\n\n', self.name, self.get_color())
        nus, flux = data
        nus_nounit = nus.value# np.array([v.value for v in nus])
        flux_nounit = flux.value#np.array([v.value for v in flux])
        if not 'color' in kwargs['plot_kwargs']:
            kwargs['plot_kwargs']['color'] = self.get_color()
        return ax.plot(nus_nounit, flux_nounit, label=kwargs['label'], **kwargs['plot_kwargs'])[0]

class SynchrotronDeltaApproxAgnPy(SynchrotronBase):
    def flux_from_nu(self, nu, electron_dist, model_params, gamma_integrate):
        return Synchrotron.evaluate_sed_flux_delta_approx(
                    nu,
                    model_params['z'],
                    model_params['d_L'],
                    model_params['delta_D'],
                    model_params['B'],
                    model_params['R_b'],
                    electron_dist
                )

class SynchrotronExactAgnPy(SynchrotronBase):
    def flux_from_nu(self, nu, electron_dist, model_params, gamma_integrate):
        return Synchrotron.evaluate_sed_flux(
                    nu,
                    model_params['z'],
                    model_params['d_L'],
                    model_params['delta_D'],
                    model_params['B'],
                    model_params['R_b'],
                    electron_dist,
                    gamma=gamma_integrate
                )[0]

class SynchrotronSelfComptonAgnPy(SynchrotronBase):
    def flux_from_nu(self, nu, electron_dist, model_params, gamma_integrate):
        return SynchrotronSelfCompton.evaluate_sed_flux(
                    nu,
                    model_params['z'],
                    model_params['d_L'],
                    model_params['delta_D'],
                    model_params['B'],
                    model_params['R_b'],
                    electron_dist,
                    gamma=gamma_integrate
                )[0]

class SynchrotronSum(EvalNode):
    def def_kwargs(self, **kwargs):
        kwargs = {
                'plot_kwargs' : {},
                'factor' : 1,
                'label' : 'syn sum'
            } | kwargs

        return kwargs

    def do(self, parent_data, common, **kwargs):
        fluxsum = u.Quantity([0] * len(parent_data[0][0])) * u.Unit("erg cm-2 s-1")
        for _, flux in parent_data:
            fluxsum += flux

        return parent_data[0][0], fluxsum * kwargs['factor']

    def plot(self, data, ax, common, **kwargs):
        #print('\n\n', self.name, self.get_color())
        nus, flux = data
        nus_nounit = nus.value# np.array([v.value for v in nus])
        flux_nounit = flux.value#np.array([v.value for v in flux])
        if not 'color' in kwargs['plot_kwargs']:
            kwargs['plot_kwargs']['color'] = self.get_color()

        return ax.plot(nus_nounit, flux_nounit, label=kwargs['label'], **kwargs['plot_kwargs'])[0]




# glue code for agnpy
class NonstaticPowerLaw(PowerLaw):
    def evaluate(self, gamma):
        return super().evaluate(gamma, *(self.parameters))
class SynchrotronCompare(SynchrotronBase):
    def def_kwargs(self, **kwargs):
        kw = super().def_kwargs(**kwargs)
        if not 'gamma_inj' in kw:
            raise ValueError('need gamma_inj for calculating comparison')

        return kw

    def do(self, parent_data, common, **kwargs):
        t, m, _, _, lims = parent_data
        k_e = np.exp(t) * kwargs['gamma_inj']**(-m) * u.Unit("cm-3")
        gamma_min, gamma_max = np.array(lims[0]) * kwargs['gamma_inj']

        print(k_e, gamma_min, gamma_max, -m)
        edist = NonstaticPowerLaw(k_e=k_e, p=-m, gamma_min=gamma_min, gamma_max=gamma_max)
        return super().do(None, common, **(kwargs | {'electron_dist': edist, 'gamma_integrate': np.logspace(np.log10(gamma_min), np.log10(gamma_max), 100)}))

class SynchrotronExactAgnPyCompare(SynchrotronCompare, SynchrotronExactAgnPy):
    pass

class SSCAgnPyCompare(SynchrotronCompare, SynchrotronSelfComptonAgnPy):
    pass



        


class SynchrotronDeltaApprox(SynchrotronBase):
    def do(self, parent_data, common, **kwargs):
        """
        Overriding because we are iterating over the momentum bins instead
        of a given nu range
        """
        mp = kwargs['model_params']
        epsilon_B = mp['B'] * constants.e.si * constants.hbar / (constants.m_e**2 * constants.c**2)
        U_B = mp['B']**2 / (8 * np.pi)

        # one way nu(p)
        nu = lambda p: (mp['delta_D'] * constants.c**2 * constants.m_e) * epsilon_prime_inv(p) / (constants.h * (1 + mp['z']))
        epsilon_prime_inv = lambda p : gamma_prime_inv(p)**2 * epsilon_B
        #gamma_prime_inv = lambda p: 1 / np.sqrt(1 + (p / (constants.m_e * constants.c))**2)
        gamma_prime_inv = lambda p: np.sqrt(1 + (p / (constants.m_e * constants.c))**2)

        #the other way p(nu)
        epsilon_prime = lambda nu : (1 + mp['z']) * constants.h * nu / (mp['delta_D'] * constants.c**2 * constants.m_e)
        gamma_prime = lambda nu : np.sqrt(epsilon_prime(nu) / epsilon_B)
        # (wrong because beta != p/mc)
        #momentum = lambda nu : constants.m_e * constants.c * sqrt(1 - gamma_prime(nu)**(-2))

        p_data, N_data, _, _ = parent_data['N_data']
        nus = []
        synchro = []
        for p, N in zip(p_data, N_data):
            this_nu = nu(p)
            nus.append(this_nu)
            flux = mp['delta_D']**4 * constants.c * constants.sigma_T * U_B * gamma_prime(this_nu)**3 * N / (6 * np.pi * mp['d_L']**2)
            synchro.append(flux)

        #return np.array(nus), np.array(flux)
        print(nus[0].unit)
        print(synchro[0].unit)
        return nus, synchro

class VLineNode(EvalNode):
    def do(self, parent_data, common, **kwargs):
        return kwargs['callback'](parent_data, common, **kwargs)

    def plot(self, data, ax, common, **kwargs):
        ax.axvline(data)

