==========================================
Implementation of the SDE integration in C
==========================================


Integration schemes
-------------------

.. doxygentypedef:: scheme_t
.. doxygenfunction:: scheme_registry_lookup
.. doxygenfunction:: scheme_euler

Propagation loop
----------------
.. doxygentypedef:: coeff_call_t
.. doxygentypedef:: boundary_call_t
.. doxygentypedef:: rng_call_t
.. doxygenfunction:: integration_loop
.. doxygenfunction:: integration_loop_p
