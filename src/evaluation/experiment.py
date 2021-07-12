import time
import logging
import numpy as np

class Experiment:
    def __init__(self, batch_cls, params, **kwargs):
        self.options = {
            'name' : ""
        }
        self.options.update(kwargs)
        self.params = params
        self.batch = batch_cls(params)

        self._finished = False
        self._states = None
        self._integrator_values = None

    def __del__(self):
        del self.batch

    @property
    def finished(self):
        return self._finished

    @property
    def states(self):
        if self._states is None:
            self._states = self.batch.states
        return self._states

    @property
    def integrator_values(self):
        if self._integrator_values is None:
            self._integrator_values = self.batch.integrator_values
        return self._integrator_values

    @property
    def name(self):
        return self.options['name']

    def __getitem__(self, index):
        return self.states[index]

    def run(self):
        logging.info("Running C++ simulation with the parameters {}".format(str(self.params)))
        start_time = time.perf_counter()

        self.batch.run()
        
        duration = time.perf_counter() - start_time
        logging.info("Finished C++ simulation in %ss", duration)

        self._finished = True

        self._states = np.array(self.batch.states)
        self._integrator_values = np.array(self.batch.integrator_values).T

    def plot(self, ax, extractors, **kwargs):
        try:
            for extractor in extractors:
                extractor.plot(self, ax, **kwargs)
        except TypeError:
            extractors.plot(self, ax, **kwargs)

class ExperimentSet:
    def __init__(self, batch_cls, params_set, **kwargs):
        self.options = {}
        self.options.update(kwargs)

        self.experiments = {}
        for name, ps in params_set.items():
            kwargs['name'] = name
            self.experiments[name] = Experiment(batch_cls, ps, **kwargs)

    def __del__(self):
        for ex in self.experiments:
            del ex

    def plot(self, ax, extractors, **kwargs):
        for experiment in self.experiments.values():
            experiment.plot(ax, extractors, **kwargs)

    def run(self):
        for ex in self.experiments.values():
            ex.run()
