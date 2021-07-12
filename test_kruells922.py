import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import time
import pickle
import argparse

from scipy.stats import linregress
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, 'lib')
from pybatch.special.kruells92 import *
from pybatch.pybreakpointstate import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


plt_dir = "out"
plt_format = "pdf"
#matplotlib.use('GTK3Agg')

# index == -1: use time
# index == 0...Ndim-1 use this dimension
def green_histograms(bin_count, states, index, end_type, value_range=None):
    states_only, weights = states
    if index == -1:
        relevant_end_values = [p.t for p in states_only if p.breakpoint_state == end_type]
        relevant_weights = [w for p, w in zip(states_only, weights) if p.breakpoint_state == end_type]
        #logging.info("using t, relevant end values = %s", len(relevant_end_values))
    else:
        relevant_end_values = [p.x[index] for p in states_only if p.breakpoint_state == end_type]
        relevant_weights = [w for p, w in zip(states_only, weights) if p.breakpoint_state == end_type]
        #logging.info("using x, relevant end values = %s", len(relevant_end_values))

    if value_range is None:
        value_range = [min(relevant_end_values)[0], max(relevant_end_values)[0]]

    print(np.array(relevant_weights).shape)
    print(np.array(relevant_end_values).T[0].shape)
    G, edges = np.histogram(np.array(relevant_end_values).T[0], bin_count, range=value_range, weights=relevant_weights, density=False)
    param = edges[:-1] + (edges[1:] - edges[:-1]) / 2

    return param, G / len(states)

def green_all_new(bin_count, states, index, x_range, T):
    s0, G0 = green_histograms(bin_count, states, -1, PyBreakpointState.LOWER, (0, T))
    s1, G1 = green_histograms(bin_count, states, -1, PyBreakpointState.UPPER, (0, T))
    st, Gt = green_histograms(bin_count, states, index, PyBreakpointState.TIME, x_range)

    green_x0 = np.array([(T - s, G) for s, G in zip(s0, G0)], dtype = np.dtype([('t', float), ('G', float)]))
    green_x1 = np.array([(T - s, G) for s, G in zip(s1, G1)], dtype = np.dtype([('t', float), ('G', float)]))
    green_t0 = np.array([t for t in zip(st, Gt)], dtype = np.dtype([('x', float), ('G', float)]))
    return green_t0, green_x0, green_x1

# 1 summand aus eq. 15 (HH Guide)
# boundary condition muss callable (float) sein
def integrate_boundary(green_values, integration_variable, boundary_condition, x, plot_filename=None):
    boundary_values = np.array([boundary_condition(x) for x in green_values[integration_variable]])
    integrand = green_values['G'] * boundary_values

    if not plot_filename is None:
        fig, axs = plt.subplots(2, figsize=(6,9))
        axs[0].set_title('Integrand values ' + x)
        axs[0].set_xlabel(integration_variable)
        axs[1].set_title('Green function values (normalized) ' + x)
        axs[1].set_xlabel(integration_variable)
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')

        bar_width = green_values[integration_variable][1] - green_values[integration_variable][0]

        axs[0].bar(green_values[integration_variable], integrand, width=bar_width, linewidth=0)
        axs[1].bar(green_values[integration_variable], green_values['G'], width=bar_width, linewidth=0)

        plt.savefig(plot_filename)

    return np.sum(integrand)

def plot_trajectory(ax, state, index):
    tvals = []
    xvals = []
    for p in state.trajectory:
        tvals.append(p.t)
        xvals.append(p.x[index])

    ax.set_xlabel("time")
    ax.set_ylabel("vector index " + str(index))
    ax.plot(tvals, xvals, marker = None, linestyle = '-')

def ppwrapper(x, T, N, param, plot_trajectories = 0, plot_index = 0):
    #logging.info("Entering C++ simulation")
    start_time = time.perf_counter()

    pyb = PyBatchKruells923(x[0], x[1], param['r_inj'], T, param['dxs'], param['Kpar'], param['r'], param['Vs'], param['dt'], param['beta_s'])
    param['x0'] = x[0]
    param['x1'] = x[1]
    param['Tmax'] = T
    #pyb = PyBatchKruells924(N, param)
    pyb.run()
    states = pyb.states()
    #integrator_values = []
    #integrator_values = np.array(pyb.integrator_values).T[0]
    #print(integrator_values)

    duration = time.perf_counter() - start_time
    logging.info("Finished C++ simulation in %ss", duration)

    plotting = False
    if plot_trajectories != 0:
        plotting = True
        fig, ax = plt.subplots(1)
        ax.set_title("Example trajectories")

    for state in states:
        if plot_trajectories > 0:
            plot_trajectories -= 1
            logging.info("Plotting a trajectory")
            plot_trajectory(ax, state, plot_index)

    if plotting:
        fig_path ="{}/particles_start{:1.1f}_i{}.{}".format(plt_dir, x[plot_index], plot_index, plt_format)
        logging.info("Saving figure in %s", fig_path)
        plt.savefig(fig_path)

    del pyb

    return states#, integrator_values

def solve_timerange(times, x0, N, param, index=0):
    states = {}
    for T in times:
        logging.info("Solving for T=%s", T)
        states[T] = ppwrapper(x0, T, N, param, plot_trajectories=10, plot_index=index)

    return states

def eval_states(states, index=0):
    histograms = {}
    for T, states_T in states.items():
        bin_count = int(len(states_T[0]) / 100)
        #print(len(states_T[0]))
        #print(bin_count)
        histograms[T] = green_histograms(bin_count, states_T, index, PyBreakpointState.TIME)

    return histograms

def clean_inf(x, y):
    rx = []
    ry = []
    for x_, y_ in zip(x, y):
        if not np.isinf(y_):
            rx.append(x_)
            ry.append(y_)

    return rx, ry

def fit_powerlaw(x, y, guess = [1, -1]):
    f = lambda x, a, q : a * x**q
    popt, _ = curve_fit(f, x, y, guess)
    return popt

def add_curve_to_plot(ax, fkt, detail = 200, label = "", xlim_plot = None, **kwargs):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    if xlim_plot is None:
        xlim_plot = xlim
    
    xfit = np.arange(xlim_plot[0], xlim_plot[1], (xlim_plot[1] - xlim_plot[0]) / detail)
    yfit = fkt(xfit)
    
    ax.plot(xfit, yfit, label = label, **kwargs)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def plot_histograms(histograms, ax, index="", powerlaw=False):
    ax.set_title("Timerange")
    ax.set_xlabel("vector index " + str(index))
    ax.set_ylabel("pseudoparticle density" + " (ln)" if index == 1 else "")
    
    global_min = np.inf
    global_max = -np.inf
    for T, (x, H) in histograms.items():
        global_min = global_min if global_min < min(x) else min(x)
        global_max = global_max if global_max > max(x) else max(x)
        if index == 1:
            if powerlaw:
                a, q = fit_powerlaw(np.exp(x), H)
            H = np.log(H)
        ax.plot(x, H, label="T=" + str(T))
        if index == 1 and powerlaw:
            add_curve_to_plot(ax, lambda x : np.log(a * np.exp(x)**q), label="Power law q = " + str(q))

    ax.set_xlim([global_min, global_max])
    ax.legend()

# loads the cache_path if existing (with pickle)
# if not, calls generator to get the result, cache it and return it
def pickle_cache(cache_path, generator, clear=False):
    if not cache_path is None and clear is False:
        try:
            with open(cache_path, mode='rb') as cachefile:
                content = pickle.load(cachefile)
        except IOError:
            logging.info("Cachefile not existing")
            content = None
    else:
        content = None

    if content is None:
        content = generator()
        try:
            with open(cache_path, mode='wb') as cachefile:
                pickle.dump(content, cachefile)
        except IOError:
            logging.error("Could not store cache")

    return content

def histograms_solve(times, x0, N, param, indizes):
    states = solve_timerange(times, x0, N, param)
    with open("tmp-cache.pickle", mode="wb") as c:
        pickle.dump(states, c)

    histograms = {}

    for i in indizes:
        histograms[i] = eval_states(states, index=i)

    return histograms



parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cache')
parser.add_argument('-r', '--regenerate', action='store_true')
args = parser.parse_args()
logging.info("Arguments: " + str(args))

param = { 'dxs' : 0.25,
          'Kpar' : 5,
          'Vs' : 1,
          'r' : 4,
          'dt' : 0.1,
          'r_inj' : 0.01,
          'beta_s' : 0,#.0001,
          'dx_inj' : 1
        }


indizes = [0, 1]
times = np.array([64, 200, 640, 1000])
#times = np.array([800])
N = 500
histograms_generator = lambda : histograms_solve(times, [0, 0], N, param, indizes)
histograms = pickle_cache(args.cache, histograms_generator, args.regenerate)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for i in indizes:
    plot_histograms(histograms[i], axs[i], index=i)
fig.savefig(plt_dir + "/timerange-both." + plt_format)

