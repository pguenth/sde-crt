from abc import ABC, abstractmethod

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

    """
    def __init__(self, experiment_call, store_opts, **kwargs):
        self.experiment_call = experiment_call
        self.store_opts = store_opts 
        self.options = {}
        self.options.update(kwargs)

        self.name = self.experiment_call.__name__ if not 'name' in store_opts else store_opts['name']
        self.path = "{}/{}.{}".format(self.store_opts["dir"], self.name, self.store_opts["format"])

    @abstractmethod
    def _plot(self, experiment):
        """
        Override this method.

        :param experiment: The experiment or experiment set to be processed
        :type experiment: :py:class:`evaluation.experiment.Experiment` or :py:class:`evaluation.experiment.ExperimentSet`

        :returns: (fig, axs)
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """
        Call to acquire the experiment or experiment set, plot it and save the plot
        *args* and *kwargs* are forwarded to the experiment callback

        :returns: The experiment or experiment set.
        """
        ex = self.experiment_call(*args, **kwargs)
        fig, axs = self._plot(ex)

        logging.info("Saving figure to {}".format(self.path))
        fig.savefig(self.path)

        return ex

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
    """
    def __init__(self, *args, **kwargs):
        self.options = {
                'subtitles' : ("", ""),
                'xlabels' : ("", ""),
                'ylabels' : ("", ""),
                'log_x' : (False, False),
                'log_y' : (False, False)
        }
        self.options.update(kwargs)
        super().__init__(*args, **self.options)
        
        if not 'title' in self.options:
            self.options['title'] = self.name

    def _plot(self, ex):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

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

        return fig, axs


class ExporterDoubleHist(ExporterDoublePlot):
    """
    Export histograms for two-dimensional problems in one figure
    
    :\*args: args of :py:class:`evaluation.exporters.ExporterDoublePlot`
    :\*\*kwargs:
        * kwargs of :py:class:`evaluation.exporters.ExporterDoublePlot`
        * kwargs of :py:class:`evaluation.extractors.HistogramExtractorFinish`

    """

    def __init__(self, *args, **kwargs):
        self.options = {
        }

        self.options.update(kwargs)
        super().__init__(*args, **self.options)

    def _plot(self, ex):
        fig, axs = super()._plot(ex)

        extractor_x = HistogramExtractorFinish(0, **(self.options | {'auto_normalize' : False}))
        extractor_p = HistogramExtractorFinish(1, **self.options)

        ex.plot(axs[0], extractor_x)
        ex.plot(axs[1], extractor_p)

        return fig, axs
