import astropy.units as u
import astropy.constants as constants
from astropy.coordinates import Distance

from agnpy import spectra

import formats
import chains
from node.nodefigure import NodeFigure
from node.cache import PickleNodeCache
from node.special import *
from node.node import *

import proplot as pplt
from matplotlib.lines import Line2D

from pybatch.special.kruells import *

# gauss/tesla equivalency
gauss_tesla_eq = (u.G, u.T, lambda x: x / np.sqrt(4 * np.pi / constants.mu0), lambda x: x * np.sqrt(4 * np.pi / constants.mu0))
cgs_gauss_unit = "cm(-1/2) g(1/2) s-1"
cgs_gauss_eq = (u.G, u.Unit(cgs_gauss_unit), lambda x: x, lambda x: x)

logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

cachedir = 'pickle/ex'
figdir = 'figures/ex'

# SI/Kruells
#def phys_k_syn_from_B(B):
#    B_si = B.to("T", equivalencies=[gauss_tesla_eq])
#    k_syn = B_si**2 / (6 * np.pi * constants.eps0) * (constants.e.si / (constants.m_e * constants.c))**4
#    return k_syn.decompose()
#
#def B_from_phys_k_syn(k_syn):
#    B_si = (constants.m_e * constants.c / constants.e.si)**2 * np.sqrt(6 * np.pi * constants.eps0 * k_syn)
#    return B_si.to("G", equivalencies=[gauss_tesla_eq])

# Gauss/Schlickeiser
def phys_k_syn_from_B(B):
    B_cgs = B.to(cgs_gauss_unit, equivalencies=[cgs_gauss_eq])
    k_syn = (constants.e.gauss**4 * B_cgs**2 / (constants.m_e**4 * constants.c**6)).decompose()
    print(k_syn)
    return k_syn

def B_from_phys_k_syn(k_syn):
    B_cgs = np.sqrt(k_syn * constants.m_e**4 * constants.c**6 / (constants.e.gauss**4))
    B = B_cgs.to(u.G, equivalencies=[cgs_gauss_eq])
    print(B)
    return B

#class PhysicalKruells(EvalNode):
#    def def_kwargs(self, **kwargs):
#        if not 'physical_params' in kwargs:
#            raise ValueError("Need physical_params keyword argument")
#
#        return kwargs
#
#    def common(self, common, **kwargs):
#        phys = kwargs['physical_params']
#        num = common['batch_param']
#
#        phys['beta_s'] = num['beta_s'] * phys['beta_0']
#        phys['q'] = num['q'] * phys['kappa_0'] / phys['beta_0']**2
#        phys['k_syn'] = num['k_syn'] * 3 * constants.c**2 * phys['beta_0']**2 / (2 * phys['kappa_0'] * phys['p_inj'])
#        phys['B'] = constants.m_e**2 * constants.c**2 / constants.e.si**2 * np.sqrt(6 * np.pi * constants.eps0 * phys['k_syn'])
#        #phys['Bgauss'] = 

# xdiff = 1.8, xadv = 1.17


class PhysicalBatchWrapper(ABC):
    """
    Imitates the interface of "PyPseudoParticleBatch"
    Override :py:meth:`get_batch_params` and :py:meth:`get_batch_cls`
    """
    def __init__(self, params=None, phys_params=None, num_params=None):
        if phys_params is None and num_params is None and not params is None:
            phys_params = params['phys']
            num_params = params['num']
        elif not (not phys_params is None and not num_params is None and params is None):
            raise ValueError("Invalid combination of arguments")

        self._batch_params = type(self).get_batch_params(phys_params, num_params)
        self._phys_params = phys_params
        self._num_parmas = num_params
        self._batch = type(self).get_batch_cls()(self._batch_params)

    @staticmethod
    @abstractmethod
    def get_batch_params(phys_params, num_params):
        pass

    @staticmethod
    @abstractmethod
    def get_batch_cls():
        pass

    @property
    def batch_params(self):
        return self._batch_params

    @property
    def phys_params(self):
        return self._phys_params

    @property
    def num_params(self):
        return self._num_params

    @property
    def integrator_values(self):
        return self._batch.integrator_values

    def run(self, *args, **kwargs):
        return self._batch.run(*args, **kwargs)

    def step_all(self, *args, **kwargs):
        return self._batch.step_all(*args, **kwargs)

    @property
    def unfinished_count(self):
        return self._batch.unfinished_count
    
    def state(self, *args, **kwargs):
        return self._batch.state(*args, **kwargs)

    @property
    def states(self):
        return self._batch.states

class PhysBatchKruells9(PhysicalBatchWrapper):
    @staticmethod
    def get_batch_cls():
        return PyBatchKruells9

    @staticmethod
    def get_batch_params(phys_param, num_param):
        p = {}

        phys_k_syn = phys_k_syn_from_B(phys_param['B'])
        p['k_syn'] = ((2 / 3 * phys_param['kappa_0'] * phys_param['p_inj'] / (constants.c**2 * phys_param['beta_0']**2) * phys_k_syn).decompose()).value
        print(p['k_syn'])

        p['beta_s'] = phys_param['beta_s'] / phys_param['beta_0']
        p['q'] = (phys_param['q'] / phys_param['kappa_0'] * phys_param['beta_0']**2).value
        p['Xsh'] = ((phys_param['Xsh'] * constants.c * phys_param['beta_0'] / phys_param['kappa_0']).decompose()).value
        t0 = phys_param['kappa_0'] / (constants.c**2 * phys_param['beta_0']**2)
        p['Tmax'] = (phys_param['Tmax'] / t0).value
        p['t_inj'] = (phys_param['t_inj'] / t0).value
        p['r'] = phys_param['r']
        p.update(num_param)

        print(p)
        return p
        
name = 'kruells9b2'

gamma_inj = 1000

param = { 'Xsh' : 1.5,
          'beta_s' : 0.9,
          'r' : 4,
          'dt' : 0.8,
          't_inj' : 0.1,
          'k_syn' : 0.005,
          'x0' : 0,
          'y0' : 1,
          'q' : 5,
          'Tmax' : 2000
    }

kappa_0 = 1e6 * u.Unit("m2 s-1")
beta_0 = 1e-6

phys_params = {
        'p_inj' : gamma_inj * constants.m_e * constants.c,
        'beta_0' : beta_0,
        'kappa_0' : kappa_0,
        'n0' : 1,
        'beta_s' : beta_0 * param['beta_s'],
        'q' : param['q'] * kappa_0 / beta_0**2,
        'Xsh' : param['Xsh'] * kappa_0 / (beta_0 * constants.c),
        'Tmax' : param['Tmax'] * kappa_0 / (beta_0 * constants.c)**2,
        't_inj' : param['t_inj'] * kappa_0 / (beta_0 * constants.c)**2,
        'r' : 4,
    }

num_params = {
        'x0' : 0,
        'y0' : 1,
        'dt' : 0.8
    }

#ksyn = [0, 0.0001, 0.0005]
#param_sets = {'k_syn={}'.format(ks) : {'param' : param | {'k_syn': ks}} for ks in ksyn}
k_syn2 = 0.0005
B2 = B_from_phys_k_syn(3 / 2 /(phys_params['kappa_0'] * phys_params['p_inj']) * (constants.c**2 * phys_params['beta_0']**2) * k_syn2)
Bs = [0 * u.G, B2]
param_sets = {'B={}'.format(B) : {'phys_params' : phys_params | {'B': B}, 'num_params' : num_params} for B in Bs}

cache = PickleNodeCache(cachedir, name)
batch = PhysBatchNode('batch', batch_cls=PhysBatchKruells9, cache=cache, ignore_cache=False)
points = PointNode('points', {'batch' : batch}, cache=cache, ignore_cache=False)

points_range = {}
for n, kw in param_sets.items():
    points_range[n] = {'points' : points.copy(n, last_kwargs=kw)}

valuesx = ValuesNode('valuesx', index=0, cache=cache, ignore_cache=False)
valuesp = ValuesNode('valuesp', index=1, cache=cache, ignore_cache=False)

histo_opts = {'bin_count' : 50, 'plot' : 'spectra', 'cache' : cache, 'ignore_cache' : False, 'label' : '$k_\\mathrm{{syn}}={k_syn}$'} 
histogramx = HistogramNode('histox', {'values' : valuesx}, log_bins=False, normalize='width', **histo_opts)
histogramp = HistogramNode('histop', {'values' : valuesp}, log_bins=True, normalize='density', **histo_opts)
powerlaw = PowerlawNode('powerlaw', {'dataset' : LimitNode('limit', parents=histogramp, lower=1, upper=200)}, plot='spectra', errors=True, ignore_cache=False)

nu_range = np.logspace(3, 19, 200) * u.Hz
d_L = 1e27 * u.cm
gamma_integrate = np.logspace(1, 9, 20)
model_params = dict(delta_D=10, z=Distance(d_L).z, d_L=d_L, R_b=1e16 * u.cm)

#def cb(model_params, batch_params):
#    k_syn = batch_params['k_syn'] * 3 * constants.c**2 / (2 * phys_param['p_inj'] * phys_param['kappa_0'])
#    B_cgs = B_from_phys_k_syn(k_syn)
#    print(B_cgs)
#    return model_params | {'B' : B_cgs}

def cb(model_params, phys_params):
    return model_params | {'B' : phys_params['B']}

radiation_params = dict(plot=True, model_params=model_params, model_params_callback=cb, nu_range=nu_range, gamma_integrate=gamma_integrate, cache=cache, ignore_cache=False)
transform = MomentumCount('mc', histogramp, plot=False, cache=cache, p_inj=phys_params['p_inj'])
synchrotronflux = SynchrotronExactAgnPy('synchro', {'N_data' : transform}, **radiation_params)
synchrotronflux_compare = SynchrotronExactAgnPyCompare('synchrocomp', powerlaw, gamma_inj=gamma_inj, plot_kwargs={'linestyle': 'dotted'}, **(radiation_params | {'ignore_cache' : False}))
sscflux_compare = SSCAgnPyCompare('ssccomp', powerlaw, gamma_inj=gamma_inj, plot_kwargs={'linestyle': 'dotted'}, **(radiation_params | {'ignore_cache' : False}))
synchrotronfluxdelta = SynchrotronDeltaApproxAgnPy('synchrodelta', {'N_data' : transform}, plot_kwargs={'linestyle': 'dashed', 'alpha': 0.6}, **radiation_params)
sscflux = SynchrotronSelfComptonAgnPy('ssc', {'N_data' : transform}, plot_kwargs={'linestyle': 'dashdot'}, **radiation_params)
#synchropeak = VLineNode('synpeak', batch, callback=lambda p, c, **kw: c['label_fmt_fields']['B'])
fluxes = NodeGroup('fluxgroup', [synchrotronflux, sscflux, synchrotronfluxdelta, sscflux_compare, synchrotronflux_compare])

histosetx = copy_to_group('groupx', histogramx, last_parents=points_range)
fluxset = copy_to_group('groupflux', fluxes, last_parents=points_range)
histosetp = NodeGroup('groupp', fluxset.search_parents_all('histop'))
powerlawset = NodeGroup('grouppl', fluxset.search_parents_all('powerlaw'))

nfig = NodeFigure(formats.singlehistSED)
#nfig.add(histosetx, 0, plot_on="spectra")
nfig.add(powerlawset, 0, plot_on="spectra")
nfig.add(fluxset, 1)
#min_flux, max_flux = np.inf, 0
#for fd in fluxset.search_parents_all("synchrodelta"):
#    vals = fd.data[1][np.logical_and(fd.data[1] != 0, np.isfinite(fd.data[1])).nonzero()]
#    if len(vals) > 0:
#        min_flux = min(min(vals), min_flux)
#        max_flux = max(max(vals), max_flux)
#    
#nfig[2].format(ylim=(min_flux.value, max_flux.value))
nfig[1].format(ylim=(1e-31, 1e-18))
pplt.rc['legend.fontsize'] = '8.0'
nfig[1].legend(ncols=1, loc='uc', handles=[
    Line2D([], [], label='Synchrotron', color='k'),
    Line2D([], [], linestyle='dashed', alpha=0.6, label='Synchrotron (delta approx.)', color='k'),
    Line2D([], [], linestyle='dashdot', label='SSC', color='k'),
    Line2D([], [], linestyle='dotted', label='comparison (assuming perfect power laws)', color='k')])
nfig.savefig(figdir + '/' + name + '.pdf')
