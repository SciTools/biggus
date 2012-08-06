# (C) British Crown Copyright 2012, Met Office
#
# This file is part of Biggus.
#
# Biggus is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Biggus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Biggus. If not, see <http://www.gnu.org/licenses/>.
"""
Virtual arrays of arbitrary size, with arithmetic and statistical
operations, and conversion to NumPy ndarrays.

Virtual arrays can be stacked to increase their dimensionality, or
tiled to increase their extent.

Includes support for easily wrapping data sources which produce NumPy
ndarray objects via slicing. For example: netcdf4python Variable
instances, and NumPy ndarray instances.

Operations which do not reduce the size of the array (e.g. element-wise
arithmetic) are performed in a lazy fashion to avoid overloading system
resources. Operations which reduce the size (e.g. taking the arithmetic
mean) will return a NumPy ndarray.

Example:
    # Wrap two large data sources (e.g. 52000 x 800 x 600).
    measured = ArrayAdapter(netcdf_var_a)
    predicted = ArrayAdapter(netcdf_var_b)

    # No actual calculations are performed here.
    error = predicted - measured

    # Calculate the mean over the first dimension, and return a real
    # NumPy ndarray. This is when the data is actually read,
    # subtracted, and the mean derived, but all in a chunk-by-chunk
    # fashion which avoids using much memory.
    mean_error = biggus.mean(error, axis=0)

"""
from abc import ABCMeta, abstractproperty, abstractmethod

import numpy


class Array(object):
    """
    A virtual array which can be sliced to create smaller virtual
    arrays, or converted to a NumPy ndarray.

    """
    __metaclass__ = ABCMeta

    @property
    def ndim(self):
        """The number of dimensions in this virtual array."""
        return len(self.shape)

    @abstractproperty
    def dtype(self):
        """The datatype of this virtual array."""

    @abstractproperty
    def shape(self):
        """The shape of the virtual array as a tuple."""

    @abstractmethod
    def __getitem__(self, keys):
        """Returns a new Array by slicing this virtual array."""

    @abstractmethod
    def ndarray(self):
        """
        Returns the NumPy ndarray instance that corresponds to this
        virtual array.

        """


class ArrayAdapter(Array):
    """
    Exposes a "concrete" array (e.g. numpy.ndarray, netCDF4.Variable)
    as an Array.

    """
    def __init__(self, concrete, keys=None):
        # concrete has:
        #   dtype
        #   ndim
        self._concrete = concrete
        result_keys = []
        if keys is not None:
            if not isinstance(keys, tuple):
                keys = (keys,)
            assert len(keys) <= concrete.ndim
            for key, size in zip(keys, concrete.shape):
                result_key = self._convert_key(key, size, 0, 1)
                result_keys.append(result_key)
        # TODO: Check if we need self._keys set to None.
        #   ... if not, remove all the `is None` checks.
        self._keys = tuple(result_keys)

    @property
    def dtype(self):
        return self._concrete.dtype

    @property
    def shape(self):
        if self._keys is None:
            shape = self._concrete.shape
        else:
            shape = _sliced_shape(self._concrete.shape, self._keys)
        return shape

    def _convert_key(self, new_key, size, start, stride):
        # Check if a key is valid for the given dimension size,
        # and map it to the concrete array, accounting for the current
        # start and stride in effect.
        if isinstance(new_key, int):
            # Is it a valid index?
            if new_key < 0:
                new_key += size
            if new_key < 0 or new_key >= size:
                raise IndexError('out of bounds')
            # If so, map it back to the concrete array.
            result_key = start + new_key * stride
        elif isinstance(new_key, slice):
            # Map the new slice back to the concrete array.
            n_start, n_stop, n_stride = new_key.indices(size)
            result_key = slice(start + stride * n_start,
                               start + stride * n_stop,
                               stride * n_stride)
        else:
            raise TypeError('invalid index: {!r}'.format(new_key))

        return result_key

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(keys) > self.ndim:
            raise IndexError('Too many keys')

        result_keys = []
        shape = list(self._concrete.shape)
        src_keys = list(self._keys or [])
        new_keys = list(keys)

        # While we still have both existing and incoming keys to
        # deal with...
        while src_keys and new_keys:
            src_size = shape.pop(0)
            src_key = src_keys.pop(0)
            if isinstance(src_key, int):
                # An integer src_key means this dimension has
                # already been sliced away - it's not visible to
                # the new keys.
                result_keys.append(src_key)
            elif isinstance(src_key, slice):
                # A slice src_key means we have to apply the new key
                # to the sliced version of the concrete dimension.
                start, stop, stride = src_key.indices(src_size)
                size = len(range(start, stop, stride))
                new_key = new_keys.pop(0)
                result_key = self._convert_key(new_key, size, start, stride)
                result_keys.append(result_key)
            else:
                raise TypeError('unsupported index type')

        # Now mop up any remaining src or new keys.
        if src_keys:
            # Any remaining src keys can just be appended.
            # (They've already been sanity checked against the
            # concrete array.)
            result_keys.extend(src_keys)
        else:
            # Any remaining new keys need to be checked against
            # the remaining dimension sizes of the concrete array.
            for new_key, size in zip(new_keys, shape):
                result_key = self._convert_key(new_key, size, 0, 1)
                result_keys.append(result_key)

        return ArrayAdapter(self._concrete, tuple(result_keys))

    def __repr__(self):
        return '<ArrayAdapter shape={} dtype={}>'.format(
            self.shape, self.dtype)

    def ndarray(self):
        if self._keys is None:
            array = self._concrete[:]
        else:
            array = self._concrete.__getitem__(self._keys)
            # We want the shape of the result to match the shape of the
            # Array, so where we've ended up with an array-scalar,
            # "inflate" it back to a 0-dimensional array.
            if array.ndim == 0:
                array = numpy.array(array)
        return array


def mean(a, axis=None):
    """
    Returns the mean of a BigArray as a NumPy ndarray.

    NB. Currently limited to axis=0.

    """
    assert axis == 0
    #   chunk_size = 2      => 54s ~ 115% CPU
    #   chunk_size = 10     => 42s ~ 105% CPU (quicker than CDO!)
    #   chunk_size = 100    => 54s
    #   chunk_size = 1000   => 63s
    size = a.shape[0]
    chunk_size = 10
    condition = threading.Condition()
    chunks = []
    def read():
        for i in range(1, size, chunk_size):
            chunk = a[i:i + chunk_size].ndarray()
            with condition:
                chunks.append(chunk)
                condition.notify()
        with condition:
            chunks.append(None)
            condition.notify()
    producer = threading.Thread(target=read)
    producer.start()

    total = a[0].ndarray()
    t = numpy.empty_like(total)
    while True:
        with condition:
            while not chunks and producer.is_alive():
                condition.wait(1)
            chunk = chunks.pop(0)
        if chunk is None:
            break
        numpy.sum(chunk, axis=0, out=t)
        total += t
    return numpy.divide(total, size, out=total)


# TODO: Test
def _sliced_shape(shape, keys):
    """
    Returns the shape that results from slicing an array of the given
    shape by the given keys.

    e.g.
        shape=(52350, 70, 90, 180)
        keys= ( 0:10,  3,  :, 2:3)
    gives:
        sliced_shape=(10, 90, 1)

    """
    sliced_shape = []
    # TODO: Watch out for more keys than shape entries.
    # TODO: Support some sort of "fancy" indexing?
    #   e.g. The first tuple in: keys=((0, 5, 12, 14), 3, :, 2:3)
    for size, key in map(None, shape, keys):
        if isinstance(key, int):
            continue
        elif isinstance(key, slice):
            size = len(range(*key.indices(size)))
            sliced_shape.append(size)
        else:
            sliced_shape.append(size)
    sliced_shape = tuple(sliced_shape)
    return sliced_shape
