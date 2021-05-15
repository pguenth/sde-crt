import matplotlib.pyplot as plt
import matplotlib
from numpy.random import default_rng
import numpy as np
from enum import Enum
import logging
import time

import sys
sys.path.insert(0, 'lib')
print(sys.path)
from pybatch.special.sourcetest import *
from pybatch.pybreakpointstate import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


plt_dir = "out"
plt_format = "png"
matplotlib.use('GTK3Agg')


class EndPosition():
    # End_type: of type EndPositionType
    # value: stores the exit time if end_type is BOUNDARY_*
    #        stores the exit position if end_type is TIME
    def __init__(self, end_type, end_value):
        self.end_type = end_type
        self.end_value = end_value

def green_histograms(bin_count, states, use_x_or_t, end_type, value_range):
    if use_x_or_t == "t":
        relevant_end_values = [p.t for p in states if p.breakpoint_state == end_type]
        #logging.info("using t, relevant end values = %s", len(relevant_end_values))
    elif use_x_or_t == "x":
        relevant_end_values = [p.x for p in states if p.breakpoint_state == end_type]
        #logging.info("using x, relevant end values = %s", len(relevant_end_values))

    G, edges = np.histogram(relevant_end_values, bin_count, range=value_range)
    param = edges[:-1] + (edges[1:] - edges[:-1]) / 2

    return param, G / len(states)

def green_all_new(bin_count, states, x_range, T):
    s0, G0 = green_histograms(bin_count, states, "x", PyBreakpointState.LOWER, (0, T))
    s1, G1 = green_histograms(bin_count, states, "x", PyBreakpointState.UPPER, (0, T))
    st, Gt = green_histograms(bin_count, states, "t", PyBreakpointState.TIME, x_range)
    
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

        bar_width = green_values[integration_variable][1] - green_values[integration_variable][0]

        axs[0].bar(green_values[integration_variable], integrand, width=bar_width, linewidth=0)
        axs[1].bar(green_values[integration_variable], green_values['G'], width=bar_width, linewidth=0)

        plt.savefig(plot_filename)

    return np.sum(integrand)

def plot_trajectory(ax, state):
    tvals = []
    xvals = []
    for p in state.trajectory:
        tvals.append(p.t)
        xvals.append(p.x[0])

    ax.plot(tvals, xvals, marker = None, linestyle = '-')

def ppwrapper(x, T, N, x_range, plot_trajectories = 0):
    #logging.info("Entering C++ simulation")
    start_time = time.perf_counter()

    pyb = PyBatchSourcetest(x, N, T, x_range[0], x_range[1])
    pyb.run()
    states = pyb.states()
    amplitudes = pyb.integrate()

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
            plot_trajectory(ax, state)

    if plotting:
        logging.info("Saving figure in %s","{}/particles_{:1.1f}.{}".format(plt_dir, x, plt_format))
        plt.savefig("{}/particles_{:1.1f}.{}".format(plt_dir, x, plt_format))

    del pyb

    return states, sum(amplitudes) / N


def solve(x, T, N, boundary_conditions, Q, plot=None):
    x_range = (-1, 1)
    bin_count = 100

    states, amplitude_correction = ppwrapper(x, T, N, x_range)

    green_t0, green_x0, green_x1 = green_all_new(bin_count, states, x_range, T)
    integrand0 = integrate_boundary(green_x0, 't', boundary_conditions['x0'], 'x=-1')#, plot + "_{}.{}".format('x0', plt_format))
    integrand1 = integrate_boundary(green_x1, 't', boundary_conditions['x1'], 'x=1')#, plot + "_{}.{}".format('x1', plt_format))
    integrandt = integrate_boundary(green_t0, 'x', boundary_conditions['T'], 't=T')#, plot + "_{}.{}".format("T", plt_format))

    #logging.info("i0: %s, i1: %s, it: %s, a: %s", integrand0, integrand1, integrandt, amplitude_correction)
    return (integrand0 + integrand1 + integrandt + amplitude_correction)

def solve_range(positions, T, N, boundary_conditions, Q):
    solutions = []
    for x in positions:
        solution = solve(x, T, N, boundary_conditions, Q, "{}/solve_{:1.1f}_{:1.1f}".format(plt_dir, T, x))
        logging.info("Solved for %s: f(%s) = %s", x, x, solution)
        solutions.append(solution)

    fig, ax = plt.subplots(1)
    ax.set_title("Solution for T={:1.1f}".format(T))
    ax.plot(positions, solutions, linestyle='none', marker='x')
    plt.savefig("{}/solve_{:1.1f}.{}".format(plt_dir, T, plt_format))


boundary_conditions_1 = {
        'x0' : lambda t : 5 / t * np.exp(-1/t-t),
        'x1' : lambda t : 1,
        'T'  : lambda x : 0
    }

boundary_conditions_2 = {
        'x0' : lambda t: 0,
        'x1' : lambda t: 0,
        'T'  : lambda x : 0 # 1 - np.abs(x)
    }

Q = lambda x : 2 if (x >= -0.5 and x <= 0) else 0

srange = np.arange(-1, 1, 0.1)
#solve_range(srange, 0.1, 5000, boundary_conditions_2, Q)
solve_range(srange, 1, 5000, boundary_conditions_2, Q)
#solve_range(srange, 1, 500, boundary_conditions_1, lambda x: 0)

    



