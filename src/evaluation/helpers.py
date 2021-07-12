import pickle
import logging

from scipy.optimize import curve_fit
import numpy as np

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

# loads the cache_path if existing (with pickle)
# if not, calls generator to get the result, cache it and return it
def pickle_cache(cache_path, generator, clear=False):
    if not cache_path is None and clear is False:
        try:
            with open(cache_path, mode='rb') as cachefile:
                logging.info("Using cached data from {}".format(cache_path))
                content = pickle.load(cachefile)
        except IOError:
            logging.info("Cachefile not existing.")
            content = None
    else:
        content = None

    if content is None:
        logging.info("No cached data found or regeneration requested. Generating data...")
        content = generator()
        try:
            with open(cache_path, mode='wb') as cachefile:
                logging.info("Storing generated data in {}".format(cache_path))
                pickle.dump(content, cachefile)
        except IOError:
            logging.error("Could not store cache")

    return content

def generate_timerange(param, times):
    pset = {}
    for t in times:
        param.update({'Tmax' : t})
        pset['T=' + str(t)] = param.copy()

    return pset

