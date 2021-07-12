import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import time

import sys
sys.path.insert(0, 'lib')
from pybatch.special.kruells1 import *
from pybatch.pybreakpointstate import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


plt_dir = "out"
plt_format = "png"
matplotlib.use('GTK3Agg')

# index == -1: use time
# index == 0...Ndim-1 use this dimension
def green_histograms(bin_count, states, index, end_type, value_range):
    if index == -1:
        relevant_end_values = [p.t for p in states if p.breakpoint_state == end_type]
        #logging.info("using t, relevant end values = %s", len(relevant_end_values))
    else:
        relevant_end_values = [p.x[index] for p in states if p.breakpoint_state == end_type]
        #logging.info("using x, relevant end values = %s", len(relevant_end_values))

    G, edges = np.histogram(relevant_end_values, bin_count, range=value_range)
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

    ax.plot(tvals, xvals, marker = None, linestyle = '-')

def ppwrapper(x, T, N, plot_trajectories = 0, plot_index = 0):
    #logging.info("Entering C++ simulation")
    start_time = time.perf_counter()

    pyb = PyBatchKruells1(x[0], x[1], N, T, 10, 0.001, 0.075, 0.025)
    pyb.run()
    states = pyb.states()

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

    return states


def solve_timerange(times, x0, N, index, bin_count = 25):
    fig, ax = plt.subplots(1)
    ax.set_title("Timerange")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for T in times:
        logging.info("Solving for T=%s", T)
        states = ppwrapper(x0, T, N, 10, index)
        st, Gt = green_histograms(bin_count, states, index, PyBreakpointState.TIME, [1, 1.5])
        ax.plot(st, Gt, label="T=" + str(T))

    ax.legend()
    ax.set_xlim(1, 2)
    fig.savefig(plt_dir + "/timerange." + plt_format)

boundary_t = lambda x : 1
    

#solve_timerange(np.array([0.64, 2.0, 6.4, 10]), [0, 1], 5000, 1)
solve_timerange(np.array([10]), [0, 1], 1000, 1)


