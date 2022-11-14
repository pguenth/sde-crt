Welcome to sde-crt's documentation!
===================================

This project aims to simplify solving of stochastic differential equations (SDEs) with the aim of solving cosmic-ray transport equations, especially in the environment of jets from active galactic nuclei.
Central to this is a general implementation of SDE solving which uses cython for shortest possible runtimes by implementing the main loops in C++.
This approach is combined with numba, which enables the user of this project to implement their SDE in pure python, which is then compiled during runtime and used by the C++ backend code.
For an introduction on how to use see :ref:`Introduction`.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :maxdepth: 4
   :caption: Contents

   introduction/index
   examples/index
   api/index
