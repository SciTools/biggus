Biggus
======

|build_status|


Virtual large arrays and lazy evaluation.

For example, we can combine multiple array data sources into a single virtual array::

    >>> first_time_series = OrthoArrayAdapter(hdf_var_a)
    >>> second_time_series = OrthoArrayAdapater(hdf_var_b)
    >>> print first_time_series.shape, second_time_series.shape
    (52000, 800, 600) (56000, 800, 600)
    >>> time_series = biggus.LinearMosaic([first_time_series, second_time_series], axis=0)
    >>> time_series
    <LinearMosaic shape=(108000, 800, 600) dtype=dtype('float32')>

*Any* biggus Array can then be indexed, independent of underlying data sources::

    >>> time_series[51999:52001, 10, 12]
    <LinearMosaic shape=(2,) dtype=dtype('float32')>
    
And an Array can be converted to a numpy ndarray on demand::

    >>> time_series[51999:52001, 10, 12].ndarray()
    array([ 0.72151309,  0.54654914], dtype=float32)


------------------------------------------
Important Note : Future development status
------------------------------------------
Biggus was originally developed primarily to provide large data handling for
`Iris <http://github.com/SciTools/iris>`_ .

For most applications it has now been surpassed by
`Dask <https://github.com/dask/dask>`_, which is used by the majority
of other projects working with virtual arrays.

Thus, **Dask is now generally preferred to Biggus**.

Iris itself now uses Dask in place of Biggus, from version 2.0 onwards.
As Biggus is little used by projects other than Iris, any future development
and support will depend on interest from other groups.

We therefore recommend that
**new code, at least, should now use Dask instead.**

Likewise, you may want to consider porting existing code to Dask, especially
if it has demanding requirements, or encounters bugs in Biggus, or requires
new features to be added.


Further reading
---------------

To get more ideas of what Biggus can do, please browse the wiki_, and its examples_.

.. _wiki: https://github.com/SciTools/biggus/wiki
.. _examples: https://github.com/SciTools/biggus/wiki/Sample-usage


Get in touch!
-------------

If you have any questions or feedback please feel free to post to the
`discussion group`_ or `raise an issue`_ on the `issue tracker`_.

.. _`discussion group`: https://groups.google.com/forum/#!forum/scitools-biggus
.. _`raise an issue`: https://github.com/SciTools/biggus/issues/new
.. _`issue tracker`: https://github.com/SciTools/biggus/issues


.. |build_status| image:: https://secure.travis-ci.org/SciTools/biggus.png
   :alt: Build Status
   :target: http://travis-ci.org/SciTools/biggus
