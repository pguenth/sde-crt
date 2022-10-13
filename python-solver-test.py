from src.sdesolver import *
from numba import njit, jit, int64, double, void, f8
from numba.typed import List
from numba.experimental import jitclass
from numba.core.typing.asnumbatype import as_numba_type
import time

@njit
def kruells94_beta(x, Xsh, a, b):
    return a - b * np.tanh(x / Xsh)

@njit
def kruells94_dbetadx(x, Xsh, b):
    return - b / (Xsh * np.cosh(x / Xsh)**2)

@njit
def kruells94_kappa_dep(x, Xsh, a, b, q):
    return q * kruells94_beta(x, Xsh, a, b)**2

@njit
def kruells94_dkappadx_dep(x, Xsh, a, b, q):
    return 2 * q * kruells94_beta(x, Xsh, a, b) * kruells94_dbetadx(x, Xsh, b)

@njit
def drift(t, x):
    # cpp: kruells_shockaccel2_drift_94_2
    Xsh = 1
    a = 1
    b = 0.5
    k_syn = 0
    q = 1

    v0 = kruells94_dkappadx_dep(x[0], Xsh, a, b, q) + kruells94_beta(x[0], Xsh, a, b)
    v1 = - (x[1]) * (kruells94_dbetadx(x[0], Xsh, b) / 3 + k_syn * x[1])
    return np.array([v0, v1])

@njit
def diffusion(t, x):
    # cpp: kruells_shockaccel2_diffusion
    Xsh = 1
    a = 1
    b = 0.5
    q = 1
    return np.array([[np.sqrt(2 * kruells94_kappa_dep(x[0], Xsh, a, b, q)), 0], [0, 0]])

@njit
def boundaries(t, x):
    if t > 100:
        return 'time'
    else:
        return None


init = [SDEPseudoParticle(0.0, np.array([0.0, 1.0])) for i in range(1)]
init_long = [SDEPseudoParticle(i * 0.1, np.array([0.0, 1.0])) for i in range(100)]
sde = SDE(2, drift, diffusion, boundaries, init)
sdesolver = SDESolver(sde_scheme_euler)
pps_finish = sdesolver.solve(sde, 0.1)

sde.initial_condition = init_long
print("start timing")
start = time.perf_counter()
res = sdesolver.solve(sde, 0.1)
end = time.perf_counter()
print("finished timing")
for pp in res:
    print(pp.t, pp.x)
print("Elapsed = {}us".format((end - start) * 1e6))

