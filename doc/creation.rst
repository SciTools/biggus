.. currentmodule:: biggus

Creating an Array
=================

To make it easy to create an Array from existing data sources, biggus
provides two adapter classes:

.. autoclass:: NumpyArrayAdapter

.. autoclass:: OrthoArrayAdapter


Combining arrays
----------------

Multiple arrays can be combined by stacking them along a new dimension,
or concatenating along an existing dimension:

.. autoclass:: ArrayStack

.. autoclass:: LinearMosaic
