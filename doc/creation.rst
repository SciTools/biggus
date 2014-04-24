.. currentmodule:: biggus

Creating an Array
=================

To make it easy to create an Array from existing data sources, biggus
provides two adapter classes:

.. autoclass:: NumpyArrayAdapter

.. autoclass:: OrthoArrayAdapter

Constants
---------

For creating an Array with the same value in every element, biggus
provides a simple class and two convenience functions which mimic NumPy.

.. autoclass:: ConstantArray

.. autofunction:: zeros
.. autofunction:: ones


Combining arrays
----------------

Multiple arrays can be combined by stacking them along a new dimension,
or concatenating along an existing dimension:

.. autoclass:: ArrayStack

.. autoclass:: LinearMosaic
