from grapheval.node import EvalNode, dict_or_list_iter, dict_or_list_map
from grapheval.cache import KwargsCacheMixin

import numpy as np
from scipy import integrate

def topt_lambda1(z, p, z0, p0, kappa1, kappa2, du):
    return  np.abs(z + z0) + \
                3 * kappa1 / du * \
                (1 + np.sqrt(kappa2 / kappa1)) * \
                np.log(p / p0)

def topt_lambda2(z, p, z0, p0, kappa1, kappa2, du):
    return  z - z0 * np.sqrt(kappa2 / kappa1) + \
                3 * kappa2 / du * \
                (1 + np.sqrt(kappa1 / kappa2)) * \
                np.log(p / p0)

def topt_N1_first(z, p, t, N0, z0, p0, du, kappa1, kappa2, u1, u2):
    if p > p0:
        l1 = topt_lambda1(z, p, z0, p0, kappa1, kappa2, du)
        return  N0 * 3 * l1 / \
                    (2 * p0**3 * du * \
                         np.sqrt(np.pi * kappa1) * \
                         t**(3/2) \
                    ) * \
                    (p0 / p)**(3/2) * \
                    np.exp(-(l1**2) / (4 * kappa1 * t) - \
                           u1 * z0 / (2 * kappa1) + \
                           u1 * z / (2 * kappa1) - \
                           u1**2 * t / (4 * kappa1)                  
                          )
    else:
        return 0
    
def topt_N2_noheaviside(z, p, t, N0, z0, p0, du, kappa1, kappa2, u1, u2):
    l2 = topt_lambda2(z, p, z0, p0, kappa1, kappa2, du)
    return N0 * 3 * l2 / \
            (2 * p0**3 * du * \
             np.sqrt(np.pi * kappa2) * \
             t**(3/2)
            ) * \
           (p0 / p)**(3/2) * \
           np.exp(-l2**2 / (4 * kappa2 * t) - \
                  u1 * z0 / (2 * kappa1) + \
                  u2 * z / (2 * kappa2) - \
                  u2**2 * t / (4 * kappa2)
                 )

def topt_N2(z, p, t, **param_set):
    if p >= param_set['p0']:
        return topt_N2_noheaviside(z, p, t, **param_set)
    else:
        return 0

def integrand_1(t, z, p, param_set):
    #                              2: U[KA94] = y^2 * F             (F[KA94] = N[T80])
    #                                  1: Density -> U = U / y
    return (p / param_set['p0'])**(2) * topt_N1_first(z, p, t, **param_set)

def Ntilde_1(z, p, T, param_set):
    return integrate.quad(
        lambda t : integrand_1(t, z, p, param_set), #(p / param_set['p0'])**2 * 
        0,
        T
    )[0]

def integrand_2(t, z, p, param_set):
    #                              2: U[KA94] = y^2 * F             (F[KA94] = N[T80])
    #                                  1: Density -> U = U / y
    return (p / param_set['p0'])**(2) * topt_N2(z, p, t, **param_set)

def Ntilde_2(z, p, T, param_set):
    return integrate.quad(
        lambda t : integrand_2(t, z, p, param_set), #(p / param_set['p0'])**2 * 
        0,
        T
    )[0]

def get_contour_set(T, x_range, y_range, detail, param_set):
    xr = np.linspace(x_range[0], x_range[1], detail)
    yr = np.logspace(y_range[0], y_range[1], num=detail)
    Z = []
    for x in xr:
        Z_ = []
        if x < 0:
            for y in yr:
                Z_.append(Ntilde_1(x, y, T, param_set))
        else:
            for y in yr:
                Z_.append(Ntilde_2(x, y, T, param_set))
            
        Z.append(Z_)
    return xr, yr, np.array(Z).T

class ToptyginContourNode(EvalNode):
    def param_set(self, params, N0):
        param_set = {
                'N0' : N0,
                'z0' : params['x0'],
                'p0' : params['y0'],
                'du' : params['beta_s'] - params['beta_s'] / params['r'],
                'kappa2' : params['q'] * (params['beta_s'] / params['r'])**2,
                'kappa1' : params['q'] * (params['beta_s'])**2,
                'u2' : params['beta_s'] / params['r'],
                'u1' : params['beta_s'],
            }
        return param_set

    def do(self, parent_data, common, params, N0, x_range, y_range, T, detail, **kwargs):
        xr = np.linspace(x_range[0], x_range[1], detail)
        yr = np.logspace(y_range[0], y_range[1], num=detail)
        Z = []
        param_set = self.param_set(params, N0)
        for x in xr:
            Z_ = []
            if x < 0:
                for y in yr:
                    Z_.append(Ntilde_1(x, y, T, param_set))
            else:
                for y in yr:
                    Z_.append(Ntilde_2(x, y, T, param_set))
                
            Z.append(Z_)
        return xr, yr, np.array(Z).T

    def plot(self, data, ax, common, levels, contour_opts={}, **kwargs):
        X, Y, Z = data
        pl = ax.contour(X, Y, np.log10(Z), levels=levels, **contour_opts)
        return pl
