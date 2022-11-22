import sdesolver as sdes

import inspect
import time

from grapheval.nodefigure import NodeFigure, NodeFigureFormat
from grapheval.cache import PickleNodeCache
from grapheval.node import *
from grapheval.graph import draw_node_chain

from src.basicnodes import *
from src.newnodes import *

from scipy import integrate
from scipy.optimize import root_scalar

import formats

from numba import njit, f8, carray
import numba

import inspect 
import logging
import ctypes
import warnings

#warnings.simplefilter("error", np.VisibleDeprecationWarning)


logging.basicConfig(level=logging.INFO, #filename='log/tests_log_{}.log'.format(sys.argv[1]),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@njit(f8(f8, f8, f8, f8))
def kruells94_beta(x, Xsh, a, b):
    return a - b * np.tanh(x / Xsh)

@njit(f8(f8, f8, f8))
def kruells94_dbetadx(x, Xsh, b):
    return - b / (Xsh * np.cosh(x / Xsh)**2)

@njit(f8(f8, f8, f8, f8, f8))
def kruells94_kappa_dep(x, Xsh, a, b, q):
    return q * kruells94_beta(x, Xsh, a, b)**2

@njit(f8(f8, f8, f8, f8, f8))
def kruells94_dkappadx_dep(x, Xsh, a, b, q):
    return 2 * q * kruells94_beta(x, Xsh, a, b) * kruells94_dbetadx(x, Xsh, b)

def drift(out, t, x, Xsh, a, b, k_syn, q):
    # cpp: kruells_shockaccel2_drift_94_2

    v0 = kruells94_dkappadx_dep(x[0], Xsh, a, b, q) + kruells94_beta(x[0], Xsh, a, b)
    v1 = - (x[1]) * (kruells94_dbetadx(x[0], Xsh, b) / 3 + k_syn * x[1])

    out[0] = v0
    out[1] = v1

def diffusion(out, t, x, Xsh, a, b, q):
    # cpp: kruells_shockaccel2_diffusion
    diffval = np.sqrt(2.0 * kruells94_kappa_dep(x[0], Xsh, a, b, q))

    # here carray is required to reshape the contiguous pointer
    out_a = carray(out, (2, 2))
    out_a[0, 0] = diffval
    out_a[1, 0] = 0
    out_a[0, 1] = 0
    out_a[1, 1] = 0

def boundaries(t, x):
    ###
    return 0
    ###

    x_a = carray(x, (2,))
    if np.abs(x_a[0]) > 0.005:
        return 1
    else:
        return 0

def disteq(eps, lastx, N, f, param):
    # trapz variant: 0.081ms/it
    deltacdf = (f(lastx, *param) + f(lastx + eps, *param)) * eps / 2
    
    # quad variant: 0.186ms/it, <1/1000 deviation from trapz
    #deltacdf_quad, _, idict = integrate.quad(f, lastx, lastx + eps, args=param, full_output=1)
    #print("quad used", idict["last"], "slices and", idict["neval"],
    #      "function calls.")
    #if not deltacdf == 0:
    #      print("It deviates by", (deltacdf - deltacdf_quad) / deltacdf_quad * 100, "% from trapz")
    return deltacdf - 1/N

def distderiv(eps, lastx, N, f, param):
    return f(lastx + eps, *param)

def nextx(lastx, N, f, param):
    root = root_scalar(disteq, x0=0, fprime=distderiv, args=(lastx, N, f, param))
    return lastx + root.root

def pullN(f, x0, N, param):
    lastx = x0
    samples = []
    for _ in range(N):
        newx = nextx(lastx, N, f, param)
        samples.append(newx)
        lastx = newx
        
    return np.array(samples)

x0 = 1
x1 = 100
alpha = 1
beta = 5
rh = 100
kwargs = {'alpha' : alpha,
        'beta' : beta,
        'rh' : rh, 
        'x0' : x0,
        'x1' : x1}
args = (alpha, beta, rh, x0, x1)

def pdf_notnorm(x, N0, alpha, beta, rh, x0, x1):
    if x < x0:
        return 0.0
    elif x < x1:
        return N0 * alpha / (beta / x - rh)**3
    else:
        return 0.0
    
pdf_notnorm_v = np.vectorize(pdf_notnorm)
integr = integrate.quad(pdf_notnorm, x0, x1, args=(1.0, alpha, beta, rh, x0, x1))[0]

def pdf_norm(x, alpha, beta, rh, x0, x1):
    return pdf_notnorm(x, 1 / integr, alpha, beta, rh, x0, x1)   

cachedir = "cache"
figdir = "figures"

name = "gcmodeling-1"

pdf_len = 1000
pdf = pullN(pdf_norm, x0, pdf_len, args)

T = 20.0
t_inj = 0.1
dt = 0.001
confine_x=100
n_particle = int(T / t_inj) * len(pdf)

init = [(int(i / len(pdf)) * t_inj, np.array([-1, pdf[i % len(pdf)]])) for i in range(n_particle)]

parameters = {
        'Xsh' : 0.001215, 
        'a' : 0.0375,
        'b' : 0.0225,
        'k_syn' : 0,
        'q' : 1,
    }


sde = sdes.SDE(2, init, drift, diffusion, boundaries)
import pickle
print(type(sde.drift))
pickle.dumps(sde.drift)
sde_pick = pickle.loads(pickle.dumps(sde))
#print(sde_pick.ndim, sde_pick.initial_condition, sde_pick.drift)

sde.set_parameters(parameters)

cache = PickleNodeCache(cachedir, name)
solvernode = SDESolverNode('solver', sde=sde, scheme=b'euler', timestep=dt, observation_times=[T], cache=cache)


valuesx = SDEValuesNode('valuesx', {'points' : solvernode[T]}, index=0, cache=cache)
valuesp = SDEValuesNode('valuesp', {'points' : solvernode[T]}, index=1, cache=cache,
        confine_range=[(0, -confine_x, confine_x)],
    )

histo_opts = {'bin_count' : 30, 'plot' : True, 'cache' : cache, 'ignore_cache' : False}
histogramx = HistogramNode('histox', {'values' : valuesx}, log_bins=False, normalize='width', **histo_opts)
histogramp = HistogramNode('histop', {'values' : valuesp}, log_bins=True, normalize='density', **histo_opts)




    #histosetx, histosetp, powerlaw = chains.get_chain_times_maxpl(NumpyBatch, cache, param, times, confine_x=0.05, bin_count=30)
    #histosetx.map_tree(lambda b : b.set(nthreads=8), "batch")

nfig = NodeFigure(formats.doublehist)
nfig.add(histogramx, 0)
nfig.add(histogramp, 1)
#nfig.add(powerlaw, 1)
nfig.savefig(figdir + '/' + name + '.pdf')



#import cProfile
#pr = cProfile.Profile()
#cProfile.run('kruells9a1_newstyle()', filename="test-cpp-propagation.perf")
