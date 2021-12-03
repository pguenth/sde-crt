from abc import ABC, abstractmethod
from collections.abc import Iterable

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'lib')
sys.path.insert(0, 'src/evaluation')
from evaluation.helpers import *
from evaluation.extractors import *

class Exporter(ABC):
    """
    Base class for combining operations to create standardized plots

    Wraps around a callable *experiment_call* which sets up and runs an 
    :py:class:`evaluation.experiment.Experiment` or :py:class:`evaluation.experiment.ExperimentSet`.
    Adds plotting and exporting functionality.

    Intended to use with :py:class:`evaluation.extractors.Extractor` classes,
    but any operation can be used. Override :py:func:`_plot` when inheriting,
    returning a tuple (fig, axs). Call the class to execute plotting and
    exporting of plots.

    :param callable experiment_call: function returning :py:class:`evaluation.experiment.Experiment` or :py:class:`evaluation.experiment.ExperimentSet`
    :param dict store_opts: Options for plot exporting
        * *dir* -- Export directory path
        * *format* -- Suffix of the requested plot format
    :\*\*kwargs:
        * *legend*
            Add a legend to every subplot if set to True


    """
    def __init__(self, experiment_call, store_opts, **kwargs):
        self.experiment_call = experiment_call
        self.store_opts = store_opts 
        self.fig = None
        self.axs = None
        self._extractors_map = []
        self._experiment = None
        self.options = {'legend' : True}
        self.options.update(kwargs)

        self.name = self.experiment_call.__name__ if not 'name' in store_opts else store_opts['name']
        self.path = "{}/{}.{}".format(self.store_opts["dir"], self.name, self.store_opts["format"])

    def attach_extractor(self, extractor, to_axes):
        self._extractors_map.append((to_axes, extractor))

    @abstractmethod
    def _prepare_plot(self, experiment):
        """
        Override this method.

        :param experiment: The experiment or experiment set to be processed
        :type experiment: :py:class:`evaluation.experiment.Experiment` or :py:class:`evaluation.experiment.ExperimentSet`

        :returns: (fig, axs)
        """
        raise NotImplementedError()

    def plot(self):
        if self._experiment is None:
            raise Exception("Experiment was not run")

        fig, axs = self._prepare_plot(self._experiment)

        self.fig = fig
        self.axs = axs

        for ax, extractor in self._extractors_map:
            extractor.plot(self._experiment, ax)

        if self.options['legend']:
            if isinstance(self.axs, Iterable):
                for ax in axs:
                    ax.legend()
            else:
                self.axs.legend()

        return self.fig, self.axs

    def store_plot(self):
        logging.info("Saving figure to {}".format(self.path))
        self.fig.savefig(self.path)

    def run_experiment(self, *args, **kwargs):
        self._experiment = self.experiment_call(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Call to acquire the experiment or experiment set, plot it and save the plot
        *args* and *kwargs* are forwarded to the experiment callback

        :returns: The experiment or experiment set.
        """
        self.run_experiment(*args, **kwargs)
        self.plot()
        self.store_plot()

        return self._experiment

class ExporterSinglePlot(Exporter):
    """
    Single Histogram
    """
    def __init__(self, *args, **kwargs):
        self.options = {
                'xlabel' : "",
                'ylabel' : "",
                'log_x' : False,
                'log_y' : False,
                'figsize' : (12, 6)
        }
        self.options.update(kwargs)
        super().__init__(*args, **self.options)
        
        if not 'title' in self.options:
            self.options['title'] = self.name

    def _prepare_plot(self, experiment):
        fig, ax = plt.subplots(1, 1, figsize=self.options['figsize'])

        if self.options['log_x']:
            ax.set_xscale('log')
        if self.options['log_y']:
            ax.set_yscale('log')
       
        ax.set_xlabel(self.options['xlabel'])
        ax.set_ylabel(self.options['ylabel'])
        ax.set_title(self.options['title'])

        return fig, ax

class ExporterHist(ExporterSinglePlot):
    def __init__(self, *args, **kwargs):
        self.options = {
        }

        self.options.update(kwargs)
        super().__init__(*args, **self.options)

    def _prepare_plot(self, experiment):
        fig, ax = super()._prepare_plot(experiment)

        self.attach_extractor(
                HistogramExtractorFinish(0, **(self.options | {'auto_normalize' : False})),
                ax
            )

        return fig, ax



class ExporterDoublePlot(Exporter):
    """
    Creates an empty two-axes plot and annotates it. Intended for inheritance by classes adding extractors
    
    :\*args: args of :py:class:`evaluation.exporters.Exporter`
    :\*\*kwargs:
        * *title*
            Title of the plot
        * *subtitles*
            2-tuple. Titles of the first and second subplot
        * *xlabels*
            2-tuple. Titles of first and second x-axis
        * *ylabels*
            2-tuple. Titles of first and second y-axis
        * *log_x*  
            2-tuple of boolean. If true the first/second x-axis' scale is set to log. *(default: (False, False))*
        * *log_y* 
            2-tuple of boolean. If true the first/second y-axis' scale is set to log. *(default: (False, False))*
        * *xlim*
            2-tuple of 2-tuple of float or None. 
        * *ylim*
            2-tuple of 2-tuple of float or None. 
    """
    def __init__(self, *args, **kwargs):
        self.options = {
                'subtitles' : ("", ""),
                'xlabels' : ("", ""),
                'ylabels' : ("", ""),
                'log_x' : (False, False),
                'log_y' : (False, False),
                'xlim' : ((None, None), (None, None)),
                'ylim' : ((None, None), (None, None))
        }
        self.options.update(kwargs)
        super().__init__(*args, **self.options)
        
        if not 'title' in self.options:
            self.options['title'] = self.name

    def _prepare_plot(self, experiment):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        for lx, ly, ax in zip(self.options['log_x'], self.options['log_y'], axs):
            if lx:
                ax.set_xscale('log')
            if ly:
                ax.set_yscale('log')
       
        fig.suptitle(self.options['title'])
        for t, x, y, ax in zip(self.options['subtitles'], self.options['xlabels'], self.options['ylabels'], axs):
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(t)

        for (xmin, xmax), (ymin, ymax), ax in zip(self.options['xlim'], self.options['ylim'], axs):
            if not xmin is None or not xmax is None:
                ax.set_xlim((xmin, xmax))
            if not ymin is None or not ymax is None:
                ax.set_ylim((ymin, ymax))

        return fig, axs


class ExporterDoubleHist(ExporterDoublePlot):
    """
    Export histograms for two-dimensional problems in one figure
    
    :\*args: args of :py:class:`evaluation.exporters.ExporterDoublePlot`
    :\*\*kwargs:
        * *log_bins*
            (log_bins_x, log_bins_p)
            Use logarithmic binning
        * kwargs of :py:class:`evaluation.exporters.ExporterDoublePlot`
        * kwargs of :py:class:`evaluation.extractors.HistogramExtractorFinish`


    """

    def __init__(self, *args, **kwargs):
        self.options = {
                'log_bins' : (False, False)
        }

        self.options.update(kwargs)
        super().__init__(*args, **self.options)

    def _get_extractors(self, experiment):
        self.ex_x = HistogramExtractorFinish(0, **(self.options | {'auto_normalize' : False, 'log_bins' : self.options['log_bins'][0]}))
        self.ex_p = HistogramExtractorFinish(1, **self.options | {'log_bins' : self.options['log_bins'][1]})
        return self.ex_x, self.ex_p


    def _prepare_plot(self, experiment):
        fig, axs = super()._prepare_plot(experiment)

        ex_x, ex_p = self._get_extractors(experiment)
        self.attach_extractor(ex_x, axs[0])
        self.attach_extractor(ex_p, axs[1])

        return fig, axs

class ExporterDoubleHistConfineP(ExporterDoubleHist):
    """
    Export histograms for two-dimensional problems in one figure
    
    :\*args: args of :py:class:`evaluation.exporters.ExporterDoubleHist`
    :\*\*kwargs:
        * kwargs of :py:class:`evaluation.exporters.ExporterDoubleHist`
        * kwargs of :py:class:`evaluation.extractors.HistogramExtractorFinish`
        * x_range_for_p : Confine PPs for the momentum plot to particles between (-x_range_for_p, x_range_for_p)

    """

    def __init__(self, *args, **kwargs):
        self.options = {
                'x_range_for_p' : 5
        }

        self.options.update(kwargs)
        super().__init__(*args, **self.options)

    def _get_extractors(self, experiment):
        ex_x, ex_p = super()._get_extractors(experiment)

        xrfp = self.options['x_range_for_p']
        ex_p.options['confinements'].append(
                (0, lambda x : (x > -xrfp and x < xrfp))
            )

        return ex_x, ex_p 

"""
    def _prepare_plot(self, experiment):
        fig, axs = super()._prepare_plot(experiment)

        ex_x, ex_p = self._get_extractors()
        self.attach_extractor(ex_x, axs[0])
        self.attach_extractor(ex_p, axs[1])

        return fig, axs
        """

class ExporterDoubleHistConfinePSingleNorm(ExporterDoubleHistConfineP):
    """
    Export histograms for two-dimensional problems in one figure
    
    :\*args: args of :py:class:`evaluation.exporters.ExporterDoubleHist`
    :\*\*kwargs:
        * kwargs of :py:class:`evaluation.exporters.ExporterDoubleHist`
        * kwargs of :py:class:`evaluation.extractors.HistogramExtractorFinish`
        * x_range_for_p : Confine PPs for the momentum plot to particles between (-x_range_for_p, x_range_for_p)

    """

    def __init__(self, *args, **kwargs):
        self.options = {
                'x_range_for_p' : 5
        }

        self.options.update(kwargs)
        super().__init__(*args, **self.options)

    def _find_max_particle_count(self, experiment, extractor):
        particle_counts = []
        for ex in experiment.experiments.values():
            particle_counts.append(extractor.particle_count(ex))
        
        return max(particle_counts)

    def _get_extractors(self, experiment):
        ex_x, ex_p = super()._get_extractors(experiment)

        norm = 1 / self._find_max_particle_count(experiment, ex_x)
        ex_x.options['manual_normalization_factor'] = norm

        return ex_x, ex_p

"""
    def _prepare_plot(self, experiment):
        fig, axs = super()._prepare_plot(experiment)


        self.attach_extractor(ex_x, axs[0])
        self.attach_extractor(ex_p, axs[1])

        return fig, axs
"""

class ExporterDoubleHistPL(ExporterDoubleHist):
    """
    Add powerlaws to the momentum space plots

    :\*args: args of :py:class:`evaluation.exporters.ExporterDoubleHist`
    :\*\*kwargs:
        * kwargs of :py:class:`evaluation.exporters.ExporterDoubleHist`
        * kwargs of :py:class:`evaluation.extractors.HistogramExtractorFinish`
        * kwargs of :py:class:`evaluation.extractors.PowerlawExtractor`
    """

    def _prepare_plot(self, experiment):
        fig, axs = super()._prepare_plot(experiment)

        self.options['ln_x'] = not self.options['log_x'][1]

        self.attach_extractor(
                PowerlawExtractor(self.ex_p, **self.options),
                axs[1]
            )

        return fig, axs
        
class ExporterDoubleHistConfinePSingleNormPL(ExporterDoubleHistConfineP):
    """
    Export histograms for two-dimensional problems in one figure
    
    :\*args: args of :py:class:`evaluation.exporters.ExporterDoubleHist`
    :\*\*kwargs:
        * kwargs of :py:class:`evaluation.exporters.ExporterDoubleHist`
        * kwargs of :py:class:`evaluation.extractors.HistogramExtractorFinish`
        * x_range_for_p : Confine PPs for the momentum plot to particles between (-x_range_for_p, x_range_for_p)

    """

    def _prepare_plot(self, experiment):
        fig, axs = super()._prepare_plot(experiment)

        ex_pl = PowerlawExtractorLinreg(self.ex_p, **self.options)
        self.attach_extractor(ex_pl, axs[1])

        return fig, axs
