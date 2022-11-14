====================
API Reference
====================

.. toctree::
   loop
   eval

SDESolver
---------

.. py:module:: sdesolver
.. autoclass:: SDE
   :members:
.. autoclass:: SDESolver
   :members:
.. autoclass:: SDECallbackBoundary
   :members:
.. autoclass:: SDECallbackCoeff
   :members:
.. py:module:: sdesolver.sdesolver
.. autoclass:: SDESolution
   :members:
   :special-members: __call__, __getitem__, __contains__
.. py:module:: sdesolver.sdecallback
.. autoclass:: SDECallbackBase
   :members:
   :special-members: __call__, __getitem__, __contains__

Methods
-------
.. autofunction:: address_as_void_pointer

Backend
-------
.. py:module:: sdesolver.loop.pyloop
.. autofunction:: py_integration_loop
