from abc import ABC, abstractmethod
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath('../lib'))
import pybatch
from pybatch.pybreakpointstate import *

class Extractor(ABC):
    """
    The interface on which all extractors are built.
    
    An extractor is a encapsulated routine that takes an experiment
    and extracts what is usually a data set and eventually plots this
    data.

    :\*\*kwargs:
        *currently there are no kwargs*
    """
    def __init__(self, **kwargs):
        self.options = {
        }
        self.options.update(kwargs)

    @abstractmethod
    def data(self, experiment):
        """
        Extract the data set from the given experiment
        
        :param experiment: The experiment to extract data from
        :type experiment: :py:class:`evaluation.experiment.Experiment`
        """

        raise NotImplementedError()

    @abstractmethod
    def plot(self, experiment, ax, **kwargs):
        """
        Plot the data of the given experiment on the given ax

        :param experiment: The experiment to extract and plot data from
        :type experiment: :py:class:`evaluation.experiment.Experiment`
        :param matplotlib.axes.Axes ax: The axes to plot on
        """

        raise NotImplementedError()
    
class TrajectoryExtractor(Extractor):
    """
    Extracts a pseudo particle trajectory

    :param index: The index of the coordinate that is extracted
    :param trajectory_number: The pseudo particle number analysed

    :\*\*kwargs:
        *currently there are no kwargs*
    """
    def __init__(self, index, trajectory_number, **kwargs):
        self.options = {
        }
        self.options.update(kwargs)
        super().__init__(**self.options)

        self.index = index
        self.trajectory_number = trajectory_number

    def data(self, experiment):
        t = experiment.states[self.trajectory_number].trajectory
        return np.array([(p.t, p.x[self.index]) for p in t]).T

    def plot(self, experiment, ax):
        t, x = self.data(experiment)
        ax.plot(t, x)
        
class MultiTrajectoryExtractor(Extractor):
    """
    Extracts multiple pseudo particle trajectories

    :param index: The index of the coordinate that is extracted
    :param count: The number of pseudo particles analysed

    :\*\*kwargs:
        *currently there are no kwargs*
    """
    def __init__(self, index, count, **kwargs):
        self.options = {
        }
        self.options.update(kwargs)
        super().__init__(**self.options)

        self.index = index
        self.count = count

    def data(self, experiment):
        trajectories = []
        for s in experiment.states:
            if len(trajectories) >= self.count:
                break

            trajectories.append(np.array([(p.t, p.x) for p in s.trajectory]).T)

        return trajectories

    def plot(self, experiment, ax):
        trajectories = self.data(experiment)
        for t, x in trajectories:
            ax.plot(t, x)

class HistogramExtractor(Extractor):
    """
    Create a histogram of the distribution of pseudo particles

    :param index: The index of the coordinate that is analysed. Other coordinates are integrated.

    :\*\*kwargs:
        * *bin_count* 
            Histogram bin count (default: None)
        * *average_bin_size* 
            The average number of pseudo particles per bin. Ignored if *bin_count* is given. (default: 100)
        * *auto_normalize*  
            Normalize the total histogram area. (default: False)
        * *use_integrator* 
            Use the given integrator (by index). (default: None)
    """
    def __init__(self, index, **kwargs):
        self.options = {
            'bin_count' : None,
            'average_bin_size' : 100,
            'auto_normalize' : False,
            'use_integrator' : None
        }
        self.options.update(kwargs)
        super().__init__(**self.options)

        self.index = index

    @staticmethod
    def _get_bin_count(options, n_states):
        if not options['bin_count'] is None:
            return int(options['bin_count'])
        else:
            c = int(n_states / options['average_bin_size'])
            return c if c != 0 else 1

    def _relevant_weights(self, experiment):
        if self.options['use_integrator'] is None:
            weights = np.array([1] * len(experiment.states))
        else:
            weights = experiment.integrator_values[self.options['use_integrator']]
        relevant_weights = [w for p, w in zip(experiment.states, weights) if p.breakpoint_state == self.options['end_state']]
        return relevant_weights

    @abstractmethod
    def _relevant_end_values(self, experiment):
        raise NotImplementedError()
        
    def data(self, experiment):
        rev = self._relevant_end_values(experiment)
        weights = self._relevant_weights(experiment)
        bin_count = type(self)._get_bin_count(self.options, len(rev))

        histogram, edges = np.histogram(np.array(rev).T[0], bin_count, weights=weights, density=self.options['auto_normalize'])
        param = edges[:-1] + (edges[1:] - edges[:-1]) / 2

        return param, histogram

    def plot(self, experiment, ax, **kwargs):
        param, histogram = self.data(experiment)
        ax.plot(param, histogram, label=experiment.name, **kwargs)

class HistogramExtractorFinish(HistogramExtractor):
    """
    Create a histogram of all pseudo particles that reached a time limit

    :\*args: args of :py:class:`evaluation.extractors.HistogramExtractor`
    :\*\*kwargs: kwargs of :py:class:`evaluation.extractors.HistogramExtractor`

    """
    def __init__(self, *args, **kwargs):
        self.options = {
            'end_state' : PyBreakpointState.TIME,
        }
        self.options.update(kwargs)
        super().__init__(*args, **self.options)

    def _relevant_end_values(self, experiment):
        return [p.x[self.index] for p in experiment.states if p.breakpoint_state == self.options['end_state']]


class HistogramExtractorSpatialBoundary(HistogramExtractor):
    """
    Create a histogram of all pseudo particles that reached a given PyBreakpointState

    :param end_state: PyBreakpointState to filter
    :\*args: args of :py:class:`evaluation.extractors.HistogramExtractor`
    :\*\*kwargs:
        * kwargs of :py:class:`evaluation.extractors.HistogramExtractor`

    """
    def __init__(self, end_state, *args, **kwargs):
        self.options = {
            'end_state' : end_state,
        }
        self.options.update(kwargs)
        super().__init__(*args, **self.options)

    def _relevant_end_values(self, experiment):
        return [p.t for p in experiment.states if p.breakpoint_state == self.options['end_state']]
