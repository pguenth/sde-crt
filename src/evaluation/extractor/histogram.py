from .extractor import *
import numpy as np

import sys
sys.path.insert(0, 'lib')
from pybatch.pybreakpointstate import *

class HistogramExtractor(Extractor):
    def __init__(self, index, **kwargs):
        self.options = {
            'bin_count' : None,
            'average_bin_size' : 100,
            'auto_normalize' : False
        }
        self.options.update(kwargs)
        super().__init__(**self.options)

        self.index = index

    @staticmethod
    def get_bin_count(options, n_states):
        if not options['bin_count'] is None:
            return int(options['bin_count'])
        else:
            c = int(n_states / options['average_bin_size'])
            return c if c != 0 else 1

    def _relevant_weights(self, experiment):
        weights = experiment.integrator_values[self.options['use_integrator']]
        relevant_weights = [w for p, w in zip(experiment.states, weights) if p.breakpoint_state == self.options['end_state']]
        return relevant_weights

    @abstractmethod
    def _relevant_end_values(self, experiment):
        raise NotImplementedError()
        
    def data(self, experiment):
        rev = self._relevant_end_values(experiment)
        weights = self._relevant_weights(experiment)
        bin_count = type(self).get_bin_count(self.options, len(rev))

        histogram, edges = np.histogram(np.array(rev).T[0], bin_count, weights=weights, density=self.options['auto_normalize'])
        param = edges[:-1] + (edges[1:] - edges[:-1]) / 2

        return param, histogram

    def plot(self, experiment, ax, **kwargs):
        param, histogram = self.data(experiment)
        ax.plot(param, histogram, label=experiment.name, **kwargs)

class HistogramExtractorFinish(HistogramExtractor):
    def __init__(self, index, **kwargs):
        self.options = {
            'end_state' : PyBreakpointState.TIME,
        }
        self.options.update(kwargs)
        super().__init__(index, **self.options)

    def _relevant_end_values(self, experiment):
        return [p.x[self.index] for p in experiment.states if p.breakpoint_state == self.options['end_state']]


class HistogramExtractorSpatialBoundary(HistogramExtractor):
    def __init__(self, index, end_state, **kwargs):
        self.options = {
            'end_state' : end_state,
        }
        self.options.update(kwargs)
        super().__init__(index, **self.options)

    def _relevant_end_values(self, experiment):
        return [p.t for p in experiment.states if p.breakpoint_state == self.options['end_state']]
