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
from .cache import FileCache
from scipy import stats
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
        return np.array([p.x for p in experiment.states if p.breakpoint_state == end_state])

class ValuesNode(EvalNode):
    """
    Reqiured parents:
        experiment

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
            
        #if index == 1:
        #    print(self.name, [p[index] for p in points])
        return np.array([p[index] for p in points])


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

    #def common(self, common, **kwargs):
    #    if not 'color' in common:
    #        return {'color' : None}

        #if 'color_cycle' in kwargs:
        #    color = next(kwargs['color_cycle'])
        #else:
        #    color = None

        #return {'color' : color}

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

        return param, histogram, errors, edges

    def plot(self, v, ax, common, **kwargs):
        param, histogram, errors, _ = v

        if ValuesNode in common['_kwargs_by_type']:
            add_fields = common['_kwargs_by_type'][ValuesNode]
        else:
            add_fields = {}

        fmt_fields = common['batch_param'] | common['label_fmt_fields'] | add_fields
        label = kwargs['label'].format(**(fmt_fields))

        if kwargs['style'] == 'step':
            lines = ax.step(param, histogram, label=label, color=self.get_color(), **kwargs['plot_kwargs'])
        elif kwargs['style'] == 'line':
            shadedata = errors if kwargs['show_errors'] else None
            lines = ax.plot(param, histogram, shadedata=shadedata, label=label, color=self.get_color(), **kwargs['plot_kwargs'])
        else:
            raise ValueError("Invalid histogram plot style used. Valid: step, line")

        self.set_color(lines[0].get_color())

        return lines[0]

class PowerlawNode(EvalNode):
    def def_kwargs(self, **kwargs):
        kwargs = {
            'ln_x' : False,
            'label' : "",
            'powerlaw_annotate' : True,
            'errors' : False,
            'plot_kwargs': {},
            'negative' : False
        } | kwargs

        return kwargs

    def do(self, parent_data, common, **kwargs):
        if kwargs['errors'] or len(parent_data['dataset']) == 4:
            x, y, errors, _ = parent_data['dataset']
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

        if kwargs['negative']:
            m *= -1

        return np.exp(t), m, np.exp(t) * dt, dm, lims

    def common(self, common, **kwargs):
        if not 'color' in common:
            return {'color' : None}

    def plot(self, data, ax, common, **kwargs):
        a, q, _, dq, lims = data
        if kwargs['ln_x']:
            func = lambda x : a * np.exp(x*q)
        else:
            func = lambda x : a * x**q


        label = kwargs['label']
        if kwargs['powerlaw_annotate']:
            label += ' $q={:.2f}\\pm {:.2f}$'.format(q, dq)

        x_plot = np.linspace(lims[0][0], lims[0][1], 100)
        y_plot = func(x_plot)
        x_plot = [x for x, y in zip(x_plot, y_plot) if y <= lims[1][1] and y >= lims[1][0]]
        y_plot = [y for y in y_plot if y <= lims[1][1] and y >= lims[1][0]]

        style = dict(linestyle='dotted', color=self.get_color()) | kwargs['plot_kwargs']

        line = ax.plot(x_plot, y_plot, label=label, **style) 

        self.set_color(line[0].get_color())

        return line[0]


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

        lines = ax.plot(x, y, bardata=bardata, label=kwargs['label'], lw=1, barlw=0.5, marker='x', capsize=1.0)
        return lines[0]

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


        print(ps)

class MomentumCount(EvalNode):
    def do(self, parent_data, common, **kwargs):
        y, U, err, edges = parent_data[0]
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
        v = np.sqrt(gamma**2 - 1) * constants.m_e * constants.c
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
                'model_params_callback' : None
            } | kwargs

        if not 'model_params' in kwargs:
            raise("Need model_params for calculating synchrotron radiation")

        return kwargs

    def do(self, parent_data, common, **kwargs):
        if kwargs['model_params_callback'] is None:
            mp = kwargs['model_params']
        else:
            mp = kwargs['model_params_callback'](kwargs['model_params'], common['batch_param'])

        nus = kwargs['nu_range']
        synchro = []
        logging.info("Calculating flux")
        for nu in nus:
            flux = self.flux_from_nu(nu, parent_data['N_data'], mp, kwargs['gamma_integrate'])
            synchro.append(flux)
        logging.info("Finished calculating flux")

        return nus, synchro

    def plot(self, data, ax, common, **kwargs):
        #print('\n\n', self.name, self.get_color())
        nus, flux = data
        nus_nounit = np.array([v.value for v in nus])
        flux_nounit = np.array([v.value for v in flux])
        return ax.plot(nus_nounit, flux_nounit, label='Synchrotron', color=self.get_color(), **kwargs['plot_kwargs'])[0]

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

class SynchrotronDeltaApproxAgnPy(SynchrotronBase):
    def flux_from_nu(self, nu, N_data, model_params, gamma_integrate):
        return Synchrotron.evaluate_sed_flux_delta_approx(
                    nu,
                    model_params['z'],
                    model_params['d_L'],
                    model_params['delta_D'],
                    model_params['B'],
                    model_params['R_b'],
                    HistogramElectronDistribution(N_data[3], N_data[1])
                )

class SynchrotronExactAgnPy(SynchrotronBase):
    def flux_from_nu(self, nu, N_data, model_params, gamma_integrate):
        return Synchrotron.evaluate_sed_flux(
                    nu,
                    model_params['z'],
                    model_params['d_L'],
                    model_params['delta_D'],
                    model_params['B'],
                    model_params['R_b'],
                    HistogramElectronDistribution(N_data[3], N_data[1]),
                    gamma=gamma_integrate
                )

class SynchrotronSelfComptonAgnPy(SynchrotronBase):
    def flux_from_nu(self, nu, N_data, model_params, gamma_integrate):
        return SynchrotronSelfCompton.evaluate_sed_flux(
                    nu,
                    model_params['z'],
                    model_params['d_L'],
                    model_params['delta_D'],
                    model_params['B'],
                    model_params['R_b'],
                    HistogramElectronDistribution(N_data[3], N_data[1]),
                    gamma=gamma_integrate
                )
