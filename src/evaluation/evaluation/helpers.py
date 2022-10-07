import logging

from functools import wraps
from scipy.optimize import curve_fit
import numpy as np

def clean_inf(x, y):
    """ 
    Remove all infinite values from a set of two arrays

    If an infinite value is encountered in one of the arrays,
    the corresponding value from the other array is also deleted.

    :param x: First array
    :param y: Second array
    :returns: `([], [])` 2-tuple of two lists that contain the cleaned data
    """
    rx = []
    ry = []
    for x_, y_ in zip(x, y):
        if not np.isinf(y_):
            rx.append(x_)
            ry.append(y_)

    return rx, ry

def fit_powerlaw(x, y, guess = [1, -1]):
    """
    Fit a power law to the data given.

    The form of the power law is a * x^q

    :param x: x data
    :param y: y data
    :param guess: *(optional)* initial guess for the parameters *a* and *q*
    :returns: 2-tuple (*a*, *q*)
    """

    f = lambda x, a, q : a * x**q
    popt, _ = curve_fit(f, x, y, guess)
    return popt

def add_curve_to_plot(ax, fkt, detail = 200, label = "", xlim_plot = None, reset_lims=True, **kwargs):
    """
    Add a function's curve to a plot

    The function is calculated with the given detail along the current x axis range
    of the plot. The values are then added to the plot without changing the x and y axis range

    :param matplotlib.axes.Axes ax: The axes to plot on
    :param callable(double) fkt: The function to plot
    :param detail: *(optional)* Count of points sampled
    :param label: *(optional)* Label of the data row
    :param xlim_plot: *(optional)* override x axis limits 
    :param \*\*kwargs: kwargs passed to `ax.plot`
    :returns: None
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    if xlim_plot is None:
        xlim_plot = xlim
    
    xfit = np.arange(xlim_plot[0], xlim_plot[1], (xlim_plot[1] - xlim_plot[0]) / detail)
    yfit = fkt(xfit)
    
    ax.plot(xfit, yfit, label = label, **kwargs)
    
    if reset_lims:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

def cached(**kwargs):
    """
    Caches the function's return value in the given file

    Returns a decorator that can be used to cache function results in the given
    file using :py:func:`pickle_cache`. If the file is not existing or `regenerate` is set
    to `True`, the function is executed, the result stored and returned.
    If the file is existing and `regenerate` is set to `False`, the cache file
    is unpickled and returned.

    :\*\*kwargs:
        * *cachedir*
            The directory to store cache files in. *(default: .)*
        * *filename*
            Cache file name. If None is given, don't cache anything. *(default: func.__name__)*
        * *regenerate*
            If set to `True`, the function is executed regardless of the file existing or not. *(default: False)*

    :returns: specialized function decorator
    """
    def cached_decorator(func):
        kwargs_c = {
                'cachedir' : '.',
                'filename' : func.__name__,
                'regenerate' : False
                } | kwargs

        if kwargs_c['filename'] is None:
            cachefile = None
        else:
            cachefile = "{}/{}.pickle".format(kwargs_c['cachedir'], kwargs_c['filename'])
        regenerate = kwargs_c['regenerate']

        @wraps(func)
        def rfunc(*args, **kwargs):
            rval = pickle_cache(cachefile, lambda : func(*args, **kwargs), regenerate)
            return rval
        return rfunc
    return cached_decorator

# loads the cache_path if existing (with pickle)
# if not, calls generator to get the result, cache it and return it
def pickle_cache(cache_path, generator, clear=False):
    """
    Load a pickled file or regenerate and store data.

    Inteded to use for enabling persistence of generated data
    with minimal additional code. See also :py:func:`cached`.
    If the file is not existing or `regenerate` is set
    to `True`, the `generator` callback is executed, the result stored and returned.
    If the file is existing and `regenerate` is set to `False`, the `cachefile`
    is unpickled and returned.

    :param cachefile: Path of file used for caching
    :param callback generator: Called if regeneration of data is neccessary
    :param regenerate: *(optional)* If set to `True`, the function is executed regardless of the file existing or not.
    :returns: Unpickled or generated data
    """
    if not cache_path is None and clear is False:
        try:
            with open(cache_path, mode='rb') as cachefile:
                logging.info("Using cached data from {}".format(cache_path))
                content = pickle.load(cachefile)
                logging.info("Cached data loaded")
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
                logging.info("Generated data is stored")
        except IOError:
            logging.error("Could not store cache")
        except TypeError:
            pass

    return content

def generate_timerange(param, times):
    """
    Generate a timerange from a param set

    The `param` dict is copied for each value in `times` and the latter is
    added to the dict under the key Tmax.

    :param dict param: The common parameters
    :param times: List of Tmax
    :return: dict
    """
    pset = {}
    for t in times:
        param.update({'Tmax' : t})
        pset['T=' + str(t)] = param.copy()

    return pset

