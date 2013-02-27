# (C) British Crown Copyright 2012 - 2013, Met Office
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
import collections

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
    def __init__(self, concrete, keys=()):
        # concrete has:
        #   dtype
        #   ndim
        self._concrete = concrete
        if not isinstance(keys, tuple):
            keys = (keys,)
        assert len(keys) <= concrete.ndim
        result_keys = []
        for axis, (key, size) in enumerate(zip(keys, concrete.shape)):
            result_key = self._cleanup_new_key(key, size, axis)
            result_keys.append(key)
        self._keys = tuple(result_keys)

    @property
    def dtype(self):
        return self._concrete.dtype

    @property
    def shape(self):
        shape = _sliced_shape(self._concrete.shape, self._keys)
        return shape

    def _cleanup_new_key(self, key, size, axis):
        """
        Return a key of type int, slice, or tuple that is guaranteed
        to be valid for the given dimension size.

        Raises IndexError/TypeError for invalid keys.

        """
        if isinstance(key, int):
            if key >= size or key < -size:
                msg = 'index {0} is out of bounds for axis {1} with' \
                      ' size {2}'.format(key, axis, size)
                raise IndexError(msg)
        elif isinstance(key, slice):
            pass
        elif isinstance(key, collections.Iterable) and \
                not isinstance(key, basestring):
            # Make sure we capture the values in case we've
            # been given a one-shot iterable, like a generator.
            key = tuple(key)
            for sub_key in key:
                if sub_key >= size or sub_key < -size:
                    msg = 'index {0} is out of bounds for axis {1}' \
                          ' with size {2}'.format(sub_key, axis, size)
                    raise IndexError(msg)
        else:
            raise TypeError('invalid key {!r}'.format(key))
        return key

    def _remap_new_key(self, indices, new_key, axis):
        """
        Return a key of type int, slice, or tuple that represents the
        combination of new_key with the given indices.

        Raises IndexError/TypeError for invalid keys.

        """
        size = len(indices)
        if isinstance(new_key, int):
            if new_key >= size or new_key < -size:
                msg = 'index {0} is out of bounds for axis {1}' \
                      ' with size {2}'.format(new_key, axis, size)
                raise IndexError(msg)
            result_key = indices[new_key]
        elif isinstance(new_key, slice):
            result_key = indices.__getitem__(new_key)
        elif isinstance(new_key, collections.Iterable) and \
                not isinstance(new_key, basestring):
            # Make sure we capture the values in case we've
            # been given a one-shot iterable, like a generator.
            new_key = tuple(new_key)
            for sub_key in new_key:
                if sub_key >= size or sub_key < -size:
                    msg = 'index {0} is out of bounds for axis {1}' \
                          ' with size {2}'.format(sub_key, axis, size)
                    raise IndexError(msg)
            result_key = tuple(indices[key] for key in new_key)
        else:
            raise TypeError('invalid key {!r}'.format(new_key))
        return result_key

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(keys) > self.ndim:
            raise IndexError('too many keys')

        result_keys = []
        shape = list(self._concrete.shape)
        src_keys = list(self._keys or [])
        new_keys = list(keys)

        # While we still have both existing and incoming keys to
        # deal with...
        axis = 0
        while src_keys and new_keys:
            src_size = shape.pop(0)
            src_key = src_keys.pop(0)
            if isinstance(src_key, int):
                # An integer src_key means this dimension has
                # already been sliced away - it's not visible to
                # the new keys.
                result_key = src_key
            elif isinstance(src_key, slice):
                # A slice src_key means we have to apply the new key
                # to the sliced version of the concrete dimension.
                start, stop, stride = src_key.indices(src_size)
                indices = tuple(range(start, stop, stride))
                new_key = new_keys.pop(0)
                result_key = self._remap_new_key(indices, new_key, axis)
            else:
                new_key = new_keys.pop(0)
                result_key = self._remap_new_key(src_key, new_key, axis)
            result_keys.append(result_key)
            axis += 1

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
                result_key = self._cleanup_new_key(new_key, size, axis)
                result_keys.append(result_key)
                axis += 1

        return ArrayAdapter(self._concrete, tuple(result_keys))

    def __repr__(self):
        return '<ArrayAdapter shape={} dtype={}>'.format(
            self.shape, self.dtype)

    def ndarray(self):
        array = self._concrete.__getitem__(self._keys)
        # We want the shape of the result to match the shape of the
        # Array, so where we've ended up with an array-scalar,
        # "inflate" it back to a 0-dimensional array.
        if array.ndim == 0:
            array = numpy.array(array)
        return array


class ArrayStack(Array):
    """
    An Array made from a homogeneous array of other Arrays.

    """
    def __init__(self, stack):
        first_array = stack.flat[0]
        item_shape = first_array.shape
        dtype = first_array.dtype
        for array in stack.flat:
            if array.shape != item_shape or array.dtype != dtype:
                raise ValueError('invalid sub-array')
        self._stack = stack
        self._item_shape = item_shape
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._stack.shape + self._item_shape

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(keys) > self.ndim:
            raise IndexError('too many keys')
        for key in keys:
            if not(isinstance(key, int) or isinstance(key, slice)):
                raise TypeError('invalid index: {!r}'.format(key))

        stack_ndim = self._stack.ndim
        stack_keys = keys[:stack_ndim]
        item_keys = keys[stack_ndim:]

        stack_shape = _sliced_shape(self._stack.shape, stack_keys)
        if stack_shape:
            stack = self._stack[stack_keys]
            # If the result was 0D, convert it back to an array.
            stack = numpy.array(stack)
            for index in numpy.ndindex(stack_shape):
                item = stack[index]
                stack[index] = item[item_keys]
            result = ArrayStack(stack)
        else:
            result = self._stack[stack_keys][item_keys]
        return result

    def __repr__(self):
        return '<ArrayStack stack_shape={} item_shape={} dtype={}>'.format(
            self._stack.shape, self._item_shape, self.dtype)

    def __setitem__(self, keys, value):
        assert len(keys) == self._stack.ndim
        for key in keys:
            assert isinstance(key, int)
        assert isinstance(value, Array), type(value)
        self._stack[keys] = value

    def ndarray(self):
        data = numpy.empty(self.shape, dtype=self.dtype)
        for index in numpy.ndindex(self._stack.shape):
            data[index] = self._stack[index].ndarray()
        return data


def mean(a, axis=None):
    """
    Returns the mean of an Array as a NumPy ndarray.

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
    for size, key in map(None, shape, keys):
        if isinstance(key, int):
            continue
        elif isinstance(key, slice):
            size = len(range(*key.indices(size)))
            sliced_shape.append(size)
        elif isinstance(key, tuple):
            sliced_shape.append(len(key))
        else:
            sliced_shape.append(size)
    sliced_shape = tuple(sliced_shape)
    return sliced_shape
