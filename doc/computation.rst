.. currentmodule:: biggus

Computations with Arrays
========================

NB. When using biggus, all computations are deferred until the results
are explicitly requested.


Statistical operations
----------------------

.. autofunction:: mean

.. autofunction:: std

.. autofunction:: var


Elementwise operations
----------------------

.. autofunction:: add

.. autofunction:: sub


Evaluation
----------

It is always possible to use the `masked_array()` and/or `ndarray()`
methods which are present on every biggus Array.

.. automethod:: Array.masked_array

.. automethod:: Array.ndarray

But for multiple expressions with shared sub-expressions it is more
efficient to request the evaluations in a single call.

.. autofunction:: ndarrays

For expressions whose results are too large to return as a numpy ndarray
it is possible to request they be sent piece-by-piece to an alternative
output, e.g. an HDF variable.

.. autofunction:: save
