import numbers
import copy

import logging 

import numpy as np
from scipy import stats, optimize

from grapheval.node import EvalNode, dict_or_list_iter

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

class HistogramCutoffNode(EvalNode):
    def do(self, parent_data, common, cutoff=0.5, **kwargs):
        from scipy.interpolate import make_interp_spline
        from scipy.optimize import root_scalar
        param, histogram, errors, edges = parent_data['histogram']
        spl = make_interp_spline(param, histogram, k=3)
        co = max(histogram) * cutoff
        root = root_scalar(lambda x : spl(x) - co, bracket=(0.1 * max(param), max(param)))
        if root.flag != "converged":
            return np.nan
        else:
            return root.root

    def plot(self, v, ax, common, **kwargs):
        return ax.axvline(x=v, color=self.get_color())





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
            'edges' : None,
            'average_bin_size' : 100,
            'bin_width' : None,
            'normalize' : 'density',
            'auto_normalize' : None,
            'manual_normalization_factor' : 1,
            'transform' : None,
            'log_bins' : False,
            'label' : '',
            'style' : 'step',
            'show_errors' : True,
            'plot_kwargs' : {},
            'hide_zeros' : False
        } | kwargs

        kwargs['plot_kwargs'] = {"linewidth" : 0.7} | kwargs['plot_kwargs']

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

    def _get_norm(self, histogram, edges, mode):
        if mode == 'density':
            db = np.array(np.diff(edges), float)
            norm = 1 / db / histogram.sum()
        elif mode == 'width':
            db = np.array(np.diff(edges), float)
            norm = 1 / db
        elif isinstance(mode, numbers.Number):
            norm = mode
        elif mode is None:
            norm = 1
        else:
            raise ArgumentError(f"normalize mode must be density, width or a number or None. It is: {mode}")

        return norm

    def do(self, parent_data, common, **kwargs):
        rev = np.array(parent_data['values'])

        if 'weights' in parent_data:
            weights = np.array(parent_data['weights'])
        else:
            weights = np.ones(len(rev))

        if len(rev.shape) > 1:
            rev_dim = np.array(rev).T[0]
        else:
            rev_dim = rev

        if not kwargs['edges'] is None:
            bins = kwargs['edges']
        elif kwargs['bin_width'] is None:
            bin_count = type(self)._get_bin_count(kwargs['bin_count'], kwargs['average_bin_size'], len(rev))
            if kwargs['log_bins']:
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
            histogram_unweighted, _ = np.histogram(rev_dim, bins=bins, density=False)
            # for error estimation, use squared weights
            histogram_squareweights, _ = np.histogram(rev_dim, bins=bins, weights=weights**2, density=False)
        except ValueError as e:
            print("verr", len(weights), len(rev_dim), "NaN: ", np.count_nonzero(np.isnan(rev_dim)))
            raise e
        #print(self.name, rev, bins, histogram)
        
        norm = self._get_norm(histogram, edges, kwargs['normalize'])

        errors = histogram * np.sqrt(histogram_unweighted) / histogram_unweighted * norm * kwargs['manual_normalization_factor']
        histogram = histogram * norm * kwargs['manual_normalization_factor']

        param = edges[:-1] + np.diff(edges) / 2

        if not kwargs['transform'] is None:
            param, histogram = kwargs['transform'](param, histogram)

        return param, histogram, errors, edges

    def plot(self, v, ax, common, **kwargs):
        if kwargs['hide_zeros']:
            param = []
            histogram = []
            errors = []
            edges = []
            for x, h, dh, e in zip(*v):
                if not h == 0:
                    param.append(x)
                    histogram.append(h)
                    errors.append(dh)
                    edges.append(e)
            param = np.array(param)
            histogram = np.array(histogram)
            errors = np.array(errors)
            edges = np.array(edges + [v[3][-1]])
        else:
            param, histogram, errors, edges = v
                

        fmt_fields = self.tree_kwargs()

        label = kwargs['label'].format(**(fmt_fields))

        if kwargs['style'] == 'step':
            c = self.get_color()
            if kwargs['show_errors']:
                hupper = histogram + errors
                hlower = histogram - errors
                ax.fill_between(param, hlower, hupper, step='mid', color=c, alpha=0.5)#, label=f"testerror_{self.name}".replace("_", " "))
            # proplot where parameter not working (see github issue)
            # using edges[1:] yields the same behaviour where='mid' should
            #print("hist step max min", min(edges[1:]), max(edges[1:]), kwargs['plot_kwargs'])
            lines = ax.step(edges[1:], histogram, label=label, color=c, **kwargs['plot_kwargs'])
        elif kwargs['style'] == 'stairs':
            lines = [ax.stairs(histogram, edges, label=label, color=self.get_color(), baseline=1e-5, **kwargs['plot_kwargs'])]
        elif kwargs['style'] == 'line':
            shadedata = errors if kwargs['show_errors'] else None
            lines = ax.plot(param, histogram, shadedata=shadedata, label=label, color=self.get_color(), **kwargs['plot_kwargs'])
        else:
            raise ValueError("Invalid histogram plot style used. Valid: step, line")

        #self.set_color(lines[0].get_color())

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
        #rev_dim = np.array(rev).T[0]

        if kwargs['log_bins'][dim]:
            bins = np.logspace(np.log10(min(rev)), np.log10(max(rev)), bin_count + 1)
        else:
            bins = np.linspace(min(rev), max(rev), bin_count + 1)

        return rev, bins

    def filter_values(self, revx, revy, weights, lims):
        xf = []
        yf = []
        ws = []
        for x_, y_, w_ in zip(revx, revy, weights):
            if lims[0][0] <= x_ and x_ <= lims[0][1] and lims[1][0] <= y_ and y_ <= lims[1][1]:
                xf.append(x_)
                yf.append(y_)
                ws.append(w_)

        return np.array(xf), np.array(yf), np.array(ws)

    def do(self, parent_data, common, **kwargs):
        if 'valuesx' in parent_data and 'valuesy' in parent_data:
            vx = parent_data['valuesx']
            vy = parent_data['valuesy']
            if not len(vx) == len(vy):
                raise ValueError("Histogram2D parents need same dimension")
        elif 'values' in parent_data:
            # assuming [(x0, x1, ...), (x0, x1, ...),...] here
            vx, vy = parent_data['values'].T
        else:
            raise ValueError("Histogram2D needs either valuesx and valuesy or values as parents")

        if 'weights' in parent_data:
            ws = parent_data['weights']
        else:
            ws = np.ones(len(vx))

        revx, revy, weights = self.filter_values(vx, vy, ws, kwargs['limits'])

        revxd, bins_x = self._prepare_onedim(revx, 0, **kwargs)
        revyd, bins_y = self._prepare_onedim(revy, 1, **kwargs)

        try:
            histogram, xedges, yedges = np.histogram2d(revxd, revyd, bins=[bins_x, bins_y], weights=weights, density=False)
        except ValueError as e:
            print("verr", len(revxd), revxd)
            raise e

        histogram = histogram.T # necessary somehow, see docs of histogram2d
        
        #! ignoring weights for now
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

        #! ignore atm
        if False and ValuesNode in common['_kwargs_by_type']:
            add_fields = common['_kwargs_by_type'][ValuesNode]
        else:
            add_fields = {}

        fmt_fields = self.tree_kwargs()
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
        if False or np.count_nonzero(y) == len(y):
            max_index = len(y)
        else:
            max_index = np.argmax(y == 0)

        x = x[:max_index]
        y = y[:max_index]
        if not errors is None:
            errors = errors[:max_index]

        if len(x) < 3:
            logging.warning("Powerlaw could not be fit, too few points")
            return np.nan, np.nan, np.inf, np.inf, ((0, 1), (0, 1))

        lims = ((min(x), max(x)), (min(y), max(y)))

        if not kwargs['ln_x']:
            x = np.log(x)

        if not errors is None:
            errors = np.abs(errors / y) # dln(x) = dx / x
        y = np.log(y)

        #if True:
        #    popt, pcov = optimize.curve_fit(lambda x, m, t : t * x**m , x, y, sigma=errors, p0=[-3, 1])
        #    m, t = popt
        #    dm, dt = np.sqrt(np.diag(pcov))
        #    print(m, t, dm, dt)
        
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
        #return t, m, np.exp(t) * dt, dm, lims

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

class CDFNode(EvalNode):
    """
    Cumulative distribution function
    """
    def plot(self, data, ax, common, **kwargs):
        param, cdf = data
        return ax.plot(param, cdf)

    def do(self, parent_data, common, **kwargs):
        param, histogram, errors, edges = parent_data
        cdf = np.cumsum(histogram)
        return param, cdf

class CCDFNode(EvalNode):
    """
    Complementary Cumulative distribution function
    """
    def plot(self, data, ax, common, **kwargs):
        param, ccdf = data
        return ax.plot(param, ccdf)

    def do(self, parent_data, common, **kwargs):
        param, histogram, errors, edges = parent_data
        ccdf = np.flip(np.cumsum(np.flip(histogram)))
        return param, ccdf

class MLEPowerlawNode(PowerlawNode):
    """
    Maximum Likelihood Estimation
    for powerlaw fit

    following: https://arxiv.org/abs/0706.1062
    which was also implemented in: https://github.com/jeffalstott/powerlaw
    
    info:
    https://stats.stackexchange.com/questions/267464/algorithms-for-weighted-maximum-likelihood-parameter-estimation
    """
    def do(self, parent_data, common, **kwargs):
        vals = np.array(parent_data['values'])

        if 'weights' in parent_data:
            weights = np.array(parent_data['weights'])
        else:
            weights = np.ones(len(vals))

        if len(vals.shape) > 1:
            vals_dim = np.array(vals).T[0]
        else:
            vals_dim = vals
        
        vmin = min(vals_dim)
        vmax = max(vals_dim)
        logs = np.log(vals_dim / vmin)
        alpha = 1 + np.sum(weights) / np.sum((weights) * np.log(vals_dim / vmin))
        a = (alpha - 1) / vmin

        dalpha = (alpha - 1) / np.sqrt(len(vals_dim)) + 1 / len(vals_dim)

        lims = ((vmin, vmax), (a * vmax**(-alpha), a * vmin**(-alpha)))
        #lims = ((min(vals_dim), max(vals_dim)), (-np.inf, np.inf))
        return a, -alpha, np.nan, dalpha, lims

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

class RealScatterNode(EvalNode):
    """
    Simple scatter plot given a parent for x and one for y data

    Parent specification:

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
        return parent_data['x'], parent_data['y']

    def plot(self, data, ax, common, **kwargs):
        x, y = data
        dy = None

        if np.all(dy is None):
            bardata = None
        elif np.any(dy is None):
            bardata = [0 if b is None else b for b in bardata]
        else:
            bardata = dy

        def_kw = dict(lw=1, barlw=0.5, marker='x', capsize=1.0)
        lines = ax.plot(x, y, bardata=bardata, label=kwargs['label'], **(def_kw | kwargs['plot_kwargs']))
        return lines[0]

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

        print("cutoffnode", self.name, x_cutoff, y_cutoff)
        return x_cutoff, y_cutoff

    def plot(self, data, ax, common, **kwargs):
        x, y = data
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

class VLineNode(EvalNode):
    def def_kwargs(self, **kwargs):
        return {'plot_kwargs' : {}} | kwargs

    def do(self, parent_data, common, **kwargs):
        # import pprint
        # pprint.pprint(common)
        # print(kwargs)
        return kwargs['callback'](parent_data, common, **kwargs)

    def plot(self, data, ax, common, **kwargs):
        kw = {'color': self.get_color()} | kwargs['plot_kwargs']
        return ax.axvline(data, **kw)


class SumHistogram(HistogramNode):
    """
    sum up a list of histograms
    """
    def do(self, parent_data, common, **kwargs):
        param = None
        histogram = None
        errors = None
        edges = None
        for p, h, e, ed in parent_data:
            if param is None:
                param = p.copy()
                histogram = h.copy()
                errors = e.copy()
                edges = ed.copy()
            else:
                if not (param == p).all():
                    raise ValueError("Param of histograms to sum should be the same")
                histogram += h
                errors += e # this is wrong
                if not (edges == ed).all():
                    raise ValueError("Edges of histograms to sum should be the same")

        norm = self._get_norm(histogram, edges, kwargs['normalize'])

        errors = errors * norm * kwargs['manual_normalization_factor']
        histogram = histogram * norm * kwargs['manual_normalization_factor']

        return param, histogram, errors, edges

class ConcatValues(EvalNode):
    """
    Concatenate a list of ValueNodes
    """
    def do(self, parent_data, common, **kwargs):
        val = np.empty(0)
        w = np.empty(0)
        for v in parent_data:
            val = np.concatenate((val, v['values']))
            w = np.concatenate((w, v['weights']))

        return {'values': val, 'weights' : w}
