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
import itertools

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

    @abstractmethod
    def masked_array(self):
        """
        Returns the NumPy MaskedArray instance that corresponds to this
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

    def __eq__(self, other):
        result = NotImplemented
        if isinstance(other, ArrayAdapter):
            result = self._concrete == other._concrete
            if isinstance(result, numpy.ndarray):
                result = numpy.all(result)
        return result

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
        return '<ArrayAdapter shape={} dtype={!r}>'.format(
            self.shape, self.dtype)

    def _apply_keys(self):
        # If we have more than one tuple as a key, then NumPy does
        # "fancy" indexing, instead of "column-based" indexing, so we
        # need to use multiple indexing operations to get the right
        # result.
        keys = self._keys
        tuple_keys = [(i, key) for i, key in enumerate(keys)
                      if isinstance(key, tuple)]
        if len(tuple_keys) > 1:
            # Since we're potentially dealing with very large datasets
            # we want to cut down the array as much as possible in the
            # first iteration.
            # But we can't reliably mix tuple keys with other tuple
            # keys or with scalar keys. So the possible first cuts are:
            #  - all scalars + all slices (iff there are any scalars)
            #  - [tuple + all slices for tuple in tuples]
            # Each possible cut will reduce the dataset to different
            # size, and *ideally* we want to choose the smallest one.

            # For now though ...
            # ... use all the non-tuple keys first (if we have any) ...
            dimensions = numpy.arange(len(keys))
            if len(tuple_keys) != len(keys):
                cut_keys = list(keys)
                for i, key in tuple_keys:
                    cut_keys[i] = slice(None)
                array = self._concrete[tuple(cut_keys)]
                is_scalar = [isinstance(key, int) for key in cut_keys]
                dimensions -= numpy.cumsum(is_scalar)
            else:
                array = self._concrete
            # ... and then do each tuple in turn.
            for i, key in tuple_keys:
                cut_keys = [slice(None)] * dimensions[i]
                cut_keys.append(key)
                array = array[tuple(cut_keys)]
        else:
            array = self._concrete.__getitem__(keys)
        return array

    def ndarray(self):
        array = self._apply_keys()
        # We want the shape of the result to match the shape of the
        # Array, so where we've ended up with an array-scalar,
        # "inflate" it back to a 0-dimensional array.
        if array.ndim == 0:
            array = numpy.array(array)
        if isinstance(array, numpy.ma.MaskedArray):
            array = array.filled()
        return array

    def masked_array(self):
        array = self._apply_keys()
        # We want the shape of the result to match the shape of the
        # Array, so where we've ended up with an array-scalar,
        # "inflate" it back to a 0-dimensional array.
        if array.ndim == 0 or not isinstance(array, numpy.ma.MaskedArray):
            array = numpy.ma.MaskedArray(array)
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
        # This weird check is safe against keys[-1] being an ndarray.
        if isinstance(keys[-1], type(Ellipsis)):
            keys = keys[:-1]
        if len(keys) > self.ndim:
            raise IndexError('too many keys')
        for key in keys:
            if not(isinstance(key, (int, slice, tuple, numpy.ndarray))):
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
        return '<ArrayStack stack_shape={} item_shape={} dtype={!r}>'.format(
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

    def masked_array(self):
        data = numpy.ma.empty(self.shape, dtype=self.dtype)
        for index in numpy.ndindex(self._stack.shape):
            masked_array = self._stack[index].masked_array()
            data[index] = masked_array
            data.fill_value = masked_array.fill_value
        return data


class LinearMosaic(Array):
    def __init__(self, tiles, axis):
        tiles = numpy.array(tiles, dtype='O', ndmin=1)
        if tiles.ndim != 1:
            raise ValueError('the tiles array must be 1-dimensional')
        first = tiles[0]
        if not(0 <= axis < first.ndim):
            msg = 'invalid axis for {0}-dimensional tiles'.format(first.ndim)
            raise ValueError(msg)
        # Make sure all the tiles are compatible
        common_shape = list(first.shape)
        common_dtype = first.dtype
        del common_shape[axis]
        for tile in tiles[1:]:
            shape = list(tile.shape)
            del shape[axis]
            if shape != common_shape:
                raise ValueError('inconsistent tile shapes')
            if tile.dtype != common_dtype:
                raise ValueError('inconsistent tile dtypes')
        self._tiles = tiles
        self._axis = axis

    @property
    def dtype(self):
        return self._tiles[0].dtype

    @property
    def shape(self):
        shape = list(self._tiles[0].shape)
        for tile in self._tiles[1:]:
            shape[self._axis] += tile.shape[self._axis]
        return tuple(shape)

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(keys) > self.ndim:
            raise IndexError('too many keys')

        axis = self._axis
        if len(keys) <= axis:
            # If there aren't enough keys to affect the tiling axis
            # then it's safe to just pass the keys to each tile.
            tile = self._tiles[0]
            tiles = [tile[keys] for tile in self._tiles]
            scalar_keys = filter(lambda key: isinstance(key, int), keys)
            result = LinearMosaic(tiles, axis - len(scalar_keys))
        else:
            axis_lengths = [tile.shape[axis] for tile in self._tiles]
            offsets = numpy.cumsum([0] + axis_lengths[:-1])
            splits = offsets - 1
            axis_key = keys[axis]
            if isinstance(axis_key, int):
                # Find the single relevant tile
                tile_index = numpy.searchsorted(splits, axis_key) - 1
                tile = self._tiles[tile_index]
                tile_indices = list(keys)
                tile_indices[axis] -= offsets[tile_index]
                result = tile[tuple(tile_indices)]
            elif isinstance(axis_key, (slice, collections.Iterable)) and \
                    not isinstance(axis_key, basestring):
                # Find the list of relevant tiles.
                # NB. If the stride is large enough, this might not be a
                # contiguous subset of self._tiles.
                if isinstance(axis_key, slice):
                    size = self.shape[axis]
                    all_axis_indices = range(*axis_key.indices(size))
                else:
                    all_axis_indices = tuple(axis_key)
                tile_indices = numpy.searchsorted(splits, all_axis_indices) - 1
                pairs = itertools.izip(all_axis_indices, tile_indices)
                i = itertools.groupby(pairs, lambda axis_tile: axis_tile[1])
                tiles = []
                tile_slice = list(keys)
                for tile_index, group_of_pairs in i:
                    axis_indices = zip(*group_of_pairs)[0]
                    tile = self._tiles[tile_index]
                    axis_indices = numpy.array(axis_indices)
                    axis_indices -= offsets[tile_index]
                    if len(axis_indices) == 1:
                        # Even if we only need one value from this tile
                        # we must preserve the axis dimension by using
                        # a slice instead of a scalar.
                        start = axis_indices[0]
                        step = 1
                        stop = start + 1
                    else:
                        start = axis_indices[0]
                        step = axis_indices[1] - start
                        stop = axis_indices[-1] + step
                    tile_slice[axis] = slice(start, stop, step)
                    tiles.append(tile[tuple(tile_slice)])
                if isinstance(axis_key, slice) and \
                        axis_key.step is not None and axis_key.step < 0:
                    tiles.reverse()
                result = LinearMosaic(tiles, axis)
            else:
                raise TypeError('invalid key {!r}'.format(axis_key))

        return result

    def ndarray(self):
        data = numpy.empty(self.shape, dtype=self.dtype)
        offset = 0
        indices = [slice(None)] * self.ndim
        axis = self._axis
        for tile in self._tiles:
            next_offset = offset + tile.shape[axis]
            indices[axis] = slice(offset, next_offset)
            data[indices] = tile.ndarray()
            offset = next_offset
        return data

    def masked_array(self):
        data = numpy.ma.empty(self.shape, dtype=self.dtype)
        offset = 0
        indices = [slice(None)] * self.ndim
        axis = self._axis
        for tile in self._tiles:
            next_offset = offset + tile.shape[axis]
            indices[axis] = slice(offset, next_offset)
            data[indices] = tile.masked_array()
            offset = next_offset
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
        elif isinstance(key, (tuple, numpy.ndarray)):
            sliced_shape.append(len(key))
        else:
            sliced_shape.append(size)
    sliced_shape = tuple(sliced_shape)
    return sliced_shape
