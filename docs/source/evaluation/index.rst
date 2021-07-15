==========
Evaluation
==========

Experiment
----------
.. py:module:: evaluation.experiment
.. autoclass:: Experiment
   :members:
.. autoclass:: ExperimentSet
   :members:

Exporter
--------
.. py:module:: evaluation.exporters
.. autoclass:: Exporter
   :members:
   :private-members: _plot
   :special-members: __call__
.. autoclass:: ExporterDoublePlot
.. autoclass:: ExporterDoubleHist
   :members:

Extractor
---------
.. py:module:: evaluation.extractors
.. autoclass:: evaluation.extractors::Extractor
   :members:
.. autoclass:: TrajectoryExtractor
   :members:
.. autoclass:: MultiTrajectoryExtractor
   :members:
.. autoclass:: HistogramExtractor
   :members:
.. autoclass:: HistogramExtractorFinish
   :members:
.. autoclass:: HistogramExtractorSpatialBoundary
   :members:

Helpers
-------
.. py:module:: evaluation.helpers
.. autofunction:: clean_inf
.. autofunction:: fit_powerlaw
.. autofunction:: add_curve_to_plot
.. autofunction:: pickle_cache
.. autofunction:: generate_timerange
.. autodecorator:: cached
