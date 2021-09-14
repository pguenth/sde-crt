import time
import logging
import numpy as np
from multiprocessing import Process
from itertools import count

class Experiment:
    _ids = count(0)

    """
    Wrapper class for PyPseudoParticleBatch class.
    
    :param batch_cls: The batch class on which this experiment is based
    :type batch_cls: *inherits* :py:class:`PyPseudoParticleBatch`
    :param dict params: Simulation parameters which *batch_cls* gets passed

    :\*\*kwargs:
        * *name* 
            The name of this experiment
    """
    def __init__(self, batch_cls, params, **kwargs):
        self.id = next(self._ids)
        self.options = {
            'name' : "#{}".format(self.id)
        }
        self.options.update(kwargs)
        self.params = params
        self.batch = batch_cls(params)

        self._finished = False
        self._states = None
        self._integrator_values = None

    def __del__(self):
        try:
            del self.batch
        except AttributeError:
            pass

    @property
    def finished(self):
        """ """
        return self._finished

    @property
    def states(self):
        """
        List of :py:class:`PyPseudoParticleState` of the experiment's particles. Data is cached.
        """
        if self._states is None:
            self._states = self.batch.states
        return self._states

    @property
    def integrator_values(self):
        """
        List of N lists of each M `double` values. N is the number of pseudo particles, M is the number of integrators.
        """
        if self._integrator_values is None:
            self._integrator_values = self.batch.integrator_values
        return self._integrator_values

    @property
    def name(self):
        """ """
        return self.options['name']

    def __getitem__(self, index):
        """
        Get the :py:class:`PyPseudoParticleState` of the particle `index`
        """
        return self.states[index]

    def run(self):
        """
        Run the C++ simulation

        :returns: `self`
        """
        logging.info("[Experiment {}] Running C++ simulation with the parameters {}".format(self.options['name'], str(self.params)))
        start_time = time.perf_counter()

        self.batch.run()
        
        duration = time.perf_counter() - start_time
        logging.info("[Experiment {}] Finished C++ simulation in %ss".format(self.options['name']), duration)

        self._finished = True

        self._states = np.array(self.batch.states)
        self._integrator_values = np.array(self.batch.integrator_values).T

        return self

    def plot(self, ax, extractors, **kwargs):
        """
        Use the given `extractors` to plot on `ax`

        :param matplotlib.axes.Axes ax: The axes to plot on.
        :param extractors: One extractor or a list of extractors to extract the data rows.
        :type extractors: :py:class:`evaluation.extractors.Extractor` or [Extractor]
        :returns: None
        """
        try:
            for extractor in extractors:
                extractor.plot(self, ax, **kwargs)
        except TypeError:
            extractors.plot(self, ax, **kwargs)

class ExperimentSet:
    """
    A set of :py:class:`evaluation.experiment.Experiment` instances.

    The instances containt the same `PyPseudoParticleBatch` backend `batch_cls` but have different
    parameters given in `params_set`.

    :param batch_cls: The batch class on which the experiments are based
    :type batch_cls: *inherits* :py:class:`PyPseudoParticleBatch`
    :param dict params_set: `name` -> `params` dictionary, containing the names and the params dictionary for each experiment

    :\*\*kwargs:
        *currently there are no kwargs*
    """

    def __init__(self, batch_cls, params_set, **kwargs):
        self.options = {}
        self.options.update(kwargs)

        self.experiments = {}
        for name, ps in params_set.items():
            kwargs['name'] = name
            self.experiments[name] = Experiment(batch_cls, ps, **kwargs)

    def __del__(self):
        for ex in self.experiments.values():
            del ex

    @property
    def finished(self):
        for ex in self.experiments.values():
            if not ex.finished:
                return False

        return True

    def plot(self, ax, extractors, **kwargs):
        """
        Plot all experiments in this set on the given `ax` with the given `extractors`.

        :param matplotlib.axes.Axes ax: The axes to plot on.
        :param extractors: One extractor or a list of extractors to extract the data rows.
        :type extractors: :py:class:`evaluation.extractors.Extractor` or [Extractor]
        :returns: None
        """
        for experiment in self.experiments.values():
            experiment.plot(ax, extractors, **kwargs)

    def run(self):
        """
        Run all experiments in this set
        
        :returns: `self`
        """

        #logging.info("Forking child processes")
        logging.info("Running experiments")
        #processes = []
        for ex in self.experiments.values():
            ex.run()
            #p = Process(target=ex.run)
            #p.start()
            #processes.append(p)
        
        logging.info("Experiments finished")
        #logging.info("Waiting for child processes to complete")
        #for p in processes:
            #p.join()
        #logging.info("All child processes joined")

        return self
