from functools import cache

import numpy as np

from grapheval.node import EvalNode

from astropy import units as u
from astropy import constants

from agnpy.synchrotron.synchrotron import Synchrotron
from agnpy.compton.synchrotron_self_compton import SynchrotronSelfCompton
from agnpy.spectra import PowerLaw

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
