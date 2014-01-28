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

All operations are performed in a lazy fashion to avoid overloading
system resources. Conversion to a concrete NumPy ndarray requires an
explicit method call.

Example:
    # Wrap two large data sources (e.g. 52000 x 800 x 600).
    measured = OrthoArrayAdapter(netcdf_var_a)
    predicted = OrthoArrayAdapter(netcdf_var_b)

    # No actual calculations are performed here.
    error = predicted - measured

    # *Appear* to calculate the mean over the first dimension, and
    # return a new biggus Array with the correct shape, etc.
    # NB. No data are read and no calculations are performed.
    mean_error = biggus.mean(error, axis=0)

    # *Actually* calculate the mean, and return a NumPy ndarray.
    # This is when the data are read, subtracted, and the mean derived,
    # but all in a chunk-by-chunk fashion which avoids using much
    # memory.
    mean_error = mean_error.ndarray()

"""
from abc import ABCMeta, abstractproperty, abstractmethod
import collections
import itertools
import threading
import Queue

import numpy as np
import numpy.ma as ma


__version__ = '0.3'


class Array(object):
    """
    A virtual array which can be sliced to create smaller virtual
    arrays, or converted to a NumPy ndarray.

    """
    __metaclass__ = ABCMeta

    @staticmethod
    def ndarrays(arrays):
        """
        Return a list of NumPy ndarray objects corresponding to the given
        biggus Array objects.

        Subclasses may override this method to provide more efficient
        implementations for their instances.

        """
        return [array.ndarray() for array in arrays]

    __hash__ = None

    def __repr__(self):
        return '<{} shape={} dtype={!r}>'.format(type(self).__name__,
                                                 self.shape, self.dtype)

    @property
    def fill_value(self):
        """The value used to fill in masked values where necessary."""
        return np.ma.empty(0, dtype=self.dtype).fill_value

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


class _ArrayAdapter(Array):
    """
    Abstract base class for exposing a "concrete" data source as a
    :class:`biggus.Array`.

    """
    def __init__(self, concrete, keys=()):
        # concrete must have:
        #   dtype
        #   shape
        self.concrete = concrete
        if not isinstance(keys, tuple):
            keys = (keys,)
        assert len(keys) <= len(concrete.shape)
        result_keys = []
        for axis, (key, size) in enumerate(zip(keys, concrete.shape)):
            result_key = self._cleanup_new_key(key, size, axis)
            result_keys.append(key)
        self._keys = tuple(result_keys)

    @property
    def dtype(self):
        return self.concrete.dtype

    @property
    def fill_value(self):
        fill_value = getattr(self.concrete, 'fill_value', None)
        if fill_value is None:
            fill_value = Array.fill_value.fget(self)
        return fill_value

    @property
    def shape(self):
        shape = _sliced_shape(self.concrete.shape, self._keys)
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
        shape = list(self.concrete.shape)
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

        return type(self)(self.concrete, tuple(result_keys))

    @abstractmethod
    def _apply_keys(self):
        pass

    def ndarray(self):
        array = self._apply_keys()
        # We want the shape of the result to match the shape of the
        # Array, so where we've ended up with an array-scalar,
        # "inflate" it back to a 0-dimensional array.
        if array.ndim == 0:
            array = np.array(array)
        if ma.isMaskedArray(array):
            array = array.filled()
        return array

    def masked_array(self):
        array = self._apply_keys()
        # We want the shape of the result to match the shape of the
        # Array, so where we've ended up with an array-scalar,
        # "inflate" it back to a 0-dimensional array.
        if array.ndim == 0 or not ma.isMaskedArray(array):
            array = ma.MaskedArray(array, fill_value=self.fill_value)
        return array


class NumpyArrayAdapter(_ArrayAdapter):
    """
    Exposes a "concrete" data source which supports NumPy "fancy
    indexing" as a :class:`biggus.Array`.

    A NumPy ndarray instance is an example suitable data source.

    NB. NumPy "fancy indexing" contrasts with orthogonal indexing which
    treats multiple iterable index keys as independent.

    """
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
            dimensions = np.arange(len(keys))
            if len(tuple_keys) != len(keys):
                cut_keys = list(keys)
                for i, key in tuple_keys:
                    cut_keys[i] = slice(None)
                array = self.concrete[tuple(cut_keys)]
                is_scalar = [isinstance(key, int) for key in cut_keys]
                dimensions -= np.cumsum(is_scalar)
            else:
                # Use ellipsis indexing to ensure we have a real ndarray
                # instance to work with. (Otherwise self.concrete would
                # need to implement `take` or `__array__`.)
                array = self.concrete[...]
            # ... and then do each tuple in turn.
            for i, key in tuple_keys:
                array = np.take(array, key, axis=dimensions[i])
        else:
            array = self.concrete.__getitem__(keys)
        return array


class OrthoArrayAdapter(_ArrayAdapter):
    """
    Exposes a "concrete" data source which supports orthogonal indexing
    as a :class:`biggus.Array`.

    Orthogonal indexing treats multiple iterable index keys as
    independent (which is also the behaviour of a :class:`biggus.Array`).

    For example::

        >>> ortho_concrete.shape
        (100, 200, 300)
        >> ortho_concrete[(0, 3, 4), :, (1, 9)].shape
        (3, 200, 2)

    A netCDF4.Variable instance is an example orthogonal concrete array.

    NB. Orthogonal indexing contrasts with NumPy "fancy indexing" where
    multiple iterable index keys are zipped together to allow the
    selection of sparse locations.

    """
    def _apply_keys(self):
        array = self.concrete.__getitem__(self._keys)
        return array


class ArrayStack(Array):
    """
    An Array made from a homogeneous array of other Arrays.

    """
    def __init__(self, stack):
        first_array = stack.flat[0]
        item_shape = first_array.shape
        dtype = first_array.dtype
        fill_value = first_array.fill_value
        for array in stack.flat:
            if (array.shape != item_shape or array.dtype != dtype or
                    array.fill_value != fill_value):
                raise ValueError('invalid sub-array')
        self._stack = stack
        self._item_shape = item_shape
        self._dtype = dtype
        self._fill_value = fill_value

    @property
    def dtype(self):
        return self._dtype

    @property
    def fill_value(self):
        return self._fill_value

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
            if not(isinstance(key, (int, slice, tuple, np.ndarray))):
                raise TypeError('invalid index: {!r}'.format(key))

        stack_ndim = self._stack.ndim
        stack_keys = keys[:stack_ndim]
        item_keys = keys[stack_ndim:]

        stack_shape = _sliced_shape(self._stack.shape, stack_keys)
        if stack_shape:
            stack = self._stack[stack_keys]
            # If the result was 0D, convert it back to an array.
            stack = np.array(stack)
            for index in np.ndindex(stack_shape):
                item = stack[index]
                stack[index] = item[item_keys]
            result = ArrayStack(stack)
        else:
            result = self._stack[stack_keys][item_keys]
        return result

    def __setitem__(self, keys, value):
        assert len(keys) == self._stack.ndim
        for key in keys:
            assert isinstance(key, int)
        assert isinstance(value, Array), type(value)
        self._stack[keys] = value

    def ndarray(self):
        data = np.empty(self.shape, dtype=self.dtype)
        for index in np.ndindex(self._stack.shape):
            data[index] = self._stack[index].ndarray()
        return data

    def masked_array(self):
        data = ma.empty(self.shape, dtype=self.dtype,
                        fill_value=self.fill_value)
        for index in np.ndindex(self._stack.shape):
            masked_array = self._stack[index].masked_array()
            data[index] = masked_array
        return data


class LinearMosaic(Array):
    def __init__(self, tiles, axis):
        tiles = np.array(tiles, dtype='O', ndmin=1)
        if tiles.ndim != 1:
            raise ValueError('the tiles array must be 1-dimensional')
        first = tiles[0]
        if not(0 <= axis < first.ndim):
            msg = 'invalid axis for {0}-dimensional tiles'.format(first.ndim)
            raise ValueError(msg)
        # Make sure all the tiles are compatible
        common_shape = list(first.shape)
        common_dtype = first.dtype
        common_fill_value = first.fill_value
        del common_shape[axis]
        for tile in tiles[1:]:
            shape = list(tile.shape)
            del shape[axis]
            if shape != common_shape:
                raise ValueError('inconsistent tile shapes')
            if tile.dtype != common_dtype:
                raise ValueError('inconsistent tile dtypes')
            if tile.fill_value != common_fill_value:
                raise ValueError('inconsistent tile fill_values')
        self._tiles = tiles
        self._axis = axis
        self._cached_shape = None

    @property
    def dtype(self):
        return self._tiles[0].dtype

    @property
    def fill_value(self):
        return self._tiles[0].fill_value

    @property
    def shape(self):
        if self._cached_shape is None:
            shape = list(self._tiles[0].shape)
            for tile in self._tiles[1:]:
                shape[self._axis] += tile.shape[self._axis]
            self._cached_shape = tuple(shape)
        return self._cached_shape

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
            offsets = np.cumsum([0] + axis_lengths[:-1])
            splits = offsets - 1
            axis_key = keys[axis]
            if isinstance(axis_key, int):
                # Find the single relevant tile
                tile_index = np.searchsorted(splits, axis_key) - 1
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
                tile_indices = np.searchsorted(splits, all_axis_indices) - 1
                pairs = itertools.izip(all_axis_indices, tile_indices)
                i = itertools.groupby(pairs, lambda axis_tile: axis_tile[1])
                tiles = []
                tile_slice = list(keys)
                for tile_index, group_of_pairs in i:
                    axis_indices = zip(*group_of_pairs)[0]
                    tile = self._tiles[tile_index]
                    axis_indices = np.array(axis_indices)
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
        data = np.empty(self.shape, dtype=self.dtype)
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
        data = ma.empty(self.shape, dtype=self.dtype,
                        fill_value=self.fill_value)
        offset = 0
        indices = [slice(None)] * self.ndim
        axis = self._axis
        for tile in self._tiles:
            next_offset = offset + tile.shape[axis]
            indices[axis] = slice(offset, next_offset)
            data[indices] = tile.masked_array()
            offset = next_offset
        return data


def ndarrays(arrays):
    """
    Return a list of NumPy ndarray objects corresponding to the given
    biggus Array objects.

    This can be more efficient (and hence faster) than converting the
    individual arrays one by one.

    """
    # Group the given Arrays by their static ndarrays() methods.
    index_array_pairs_by_func = {}
    for i, array in enumerate(arrays):
        index_array_pairs = index_array_pairs_by_func.setdefault(
            array.ndarrays, [])
        index_array_pairs.append((i, array))
    # Call each static ndarrays() method and compile the results.
    all_results = [None] * len(arrays)
    for func, index_array_pairs in index_array_pairs_by_func.iteritems():
        indices = [index for index, array in index_array_pairs]
        results = func([array for index, array in index_array_pairs])
        for i, ndarray in zip(indices, results):
            all_results[i] = ndarray
    return all_results


MAX_CHUNK_SIZE = 1024 * 1024


def _all_slices(array):
    # Return the slices for each dimension which ensure complete
    # coverage by chunks no larger than MAX_CHUNK_SIZE.
    # e.g. For a float32 array of shape (100, 768, 1024) the slices are:
    #   (0, 1, 2, ..., 99),
    #   (slice(0, 256), slice(256, 512), slice(512, 768)),
    #   (slice(None)
    nbytes = array.dtype.itemsize
    all_slices = []
    for i, size in reversed(list(enumerate(array.shape))):
        if size * nbytes <= MAX_CHUNK_SIZE:
            slices = (slice(None),)
        elif nbytes > MAX_CHUNK_SIZE:
            slices = range(size)
        else:
            step = MAX_CHUNK_SIZE / nbytes
            slices = []
            for start in range(0, size, step):
                slices.append(slice(start, start + step))
        nbytes *= size
        all_slices.insert(0, slices)
    return all_slices


def save(sources, targets):
    """
    Save the numeric results of each source into its corresponding target.

    """
    # TODO: Remove restriction
    assert len(sources) == 1 and len(targets) == 1
    array = sources[0]
    target = targets[0]

    # Request bitesize pieces of the source and assign them to the
    # target.
    # NB. This algorithm does not use the minimal number of chunks.
    #   e.g. If the second dimension could be sliced as 0:99, 99:100
    #       then clearly the first dimension would have to be single
    #       slices for the 0:99 case, but could be bigger slices for the
    #       99:100 case.
    # It's not yet clear if this really matters.
    all_slices = _all_slices(array)
    for index in np.ndindex(*[len(slices) for slices in all_slices]):
        keys = tuple(slices[i] for slices, i in zip(all_slices, index))
        target[keys] = array[keys].ndarray()


class _ChunkHandler(object):
    __metaclass__ = ABCMeta

    def __init__(self, array, axis, kwargs):
        self.array = array
        self.axis = axis
        self.kwargs = kwargs

    @abstractmethod
    def bootstrap(self):
        pass

    @abstractmethod
    def add_chunk(self, chunk):
        pass

    @abstractmethod
    def result(self):
        pass


class _Mean(_ChunkHandler):
    def bootstrap(self):
        first_slice = self.array[0].ndarray()
        self.running_total = np.array(first_slice)
        self.t = np.empty_like(first_slice)

    def add_chunk(self, chunk):
        np.sum(chunk, axis=self.axis, out=self.t)
        self.running_total += self.t

    def result(self):
        self.running_total /= self.array.shape[0]
        return self.running_total


class _Std(_ChunkHandler):
    # The algorithm used here preserves numerical accuracy whilst only
    # requiring a single pass, and is taken from:
    # Welford, BP (August 1962). "Note on a Method for Calculating
    # Corrected Sums of Squares and Products".
    # Technometrics 4 (3): 419-420.
    # http://zach.in.tu-clausthal.de/teaching/info_literatur/Welford.pdf

    def bootstrap(self):
        first_slice = self.array[0].ndarray()
        self.a = np.array(first_slice)
        self.q = np.zeros_like(first_slice)
        self.t = np.empty_like(first_slice)
        self.k = 1

    def add_chunk(self, chunk):
        chunk = np.rollaxis(chunk, self.axis)
        for slice in chunk:
            self.k += 1

            # Compute A(k)
            np.subtract(slice, self.a, out=self.t)
            self.t *= 1. / self.k
            self.a += self.t

            # Compute Q(k)
            self.t *= self.t
            self.t *= self.k * (self.k - 1)
            self.q += self.t

    def result(self):
        assert self.k == self.array.shape[self.axis]
        self.q /= (self.k - self.kwargs['ddof'])
        result = np.sqrt(self.q)
        return result


class _Aggregation(Array):
    @staticmethod
    def ndarrays(arrays):
        """
        Return a list of NumPy ndarray objects corresponding to the given
        biggus _Aggregation objects.

        """
        assert all(isinstance(array, _Aggregation) for array in arrays)

        # Group the given Arrays by their sources.
        index_array_pairs_by_source = {}
        for i, array in enumerate(arrays):
            index_array_pairs = index_array_pairs_by_source.setdefault(
                id(array._array), [])
            index_array_pairs.append((i, array))
        all_results = [None] * len(arrays)
        for index_array_pairs in index_array_pairs_by_source.itervalues():
            indices = [index for index, array in index_array_pairs]
            arrays = [array for index, array in index_array_pairs]
            results = _Aggregation._ndarrays_common_source(arrays)
            for i, ndarray in zip(indices, results):
                all_results[i] = ndarray
        return all_results

    @staticmethod
    def _ndarrays_common_source(arrays):
        chunk_handlers = [array.chunk_handler() for array in arrays]
        for chunk_handler in chunk_handlers:
            chunk_handler.bootstrap()

        def meta_chunk_handler(chunk):
            for chunk_handler in chunk_handlers:
                chunk_handler.add_chunk(chunk)

        src_array = arrays[0]._array
        _process_chunks(src_array, meta_chunk_handler)

        results = [chunk_handler.result() for chunk_handler in chunk_handlers]
        return results

    def __init__(self, array, axis, chunk_handler_class, kwargs):
        self._array = array
        self._axis = axis
        self._chunk_handler_class = chunk_handler_class
        self._kwargs = kwargs

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def shape(self):
        shape = list(self._array.shape)
        del shape[self._axis]
        return tuple(shape)

    def __getitem__(self, keys):
        assert self._axis == 0
        if not isinstance(keys, tuple):
            keys = (keys,)
        keys = (slice(None),) + keys
        return _Aggregation(self._array[keys], self._axis,
                            self._chunk_handler_class, self._kwargs)

    def ndarray(self):
        chunk_handler = self.chunk_handler()
        chunk_handler.bootstrap()
        _process_chunks(self._array, chunk_handler.add_chunk)
        return chunk_handler.result()

    def masked_array(self):
        raise RuntimeError()

    def chunk_handler(self):
        return self._chunk_handler_class(self._array, self._axis, self._kwargs)


def mean(a, axis=None):
    """
    Returns the mean of an Array as another Array.

    NB. Currently limited to axis=0.

    """
    assert axis == 0
    return _Aggregation(a, axis, _Mean, {})


def std(a, axis=None, ddof=0):
    """
    Return the mean of an Array as another Array.

    NB. Currently limited to axis=0.

    """
    assert axis == 0
    return _Aggregation(a, axis, _Std, {'ddof': ddof})


class _Elementwise(Array):
    def __init__(self, array1, array2, numpy_op, ma_op):
        # TODO: Broadcasting
        assert array1.shape == array2.shape
        # TODO: Type-promotion
        assert array1.dtype == array2.dtype
        self._array1 = array1
        self._array2 = array2
        self._numpy_op = numpy_op
        self._ma_op = ma_op

    @property
    def dtype(self):
        return self._array1.dtype

    @property
    def shape(self):
        return self._array1.shape

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        return _Elementwise(self._array1[keys], self._array2[keys],
                            self._numpy_op, self._ma_op)

    def _calc(self, op):
        operands = (self._array1, self._array2)
        np_operands = ndarrays(operands)
        result = op(*np_operands)
        return result

    def ndarray(self):
        result = self._calc(self._numpy_op)
        return result

    def masked_array(self):
        result = self._calc(self._ma_op)
        return result


def add(a, b):
    """
    Return the elementwise evaluation of `a + b` as another Array.

    """
    return _Elementwise(a, b, np.add, np.ma.add)


def sub(a, b):
    """
    Return the elementwise evaluation of `a - b` as another Array.

    """
    return _Elementwise(a, b, np.subtract, np.ma.subtract)


def _process_chunks(array, chunk_handler):
    #   chunk_size = 2      => 54s ~ 115% CPU
    #   chunk_size = 10     => 42s ~ 105% CPU (quicker than CDO!)
    #   chunk_size = 100    => 54s
    #   chunk_size = 1000   => 63s
    size = array.shape[0]
    chunk_size = 10
    chunks = Queue.Queue(maxsize=3)

    def worker():
        while True:
            chunk = chunks.get()
            chunk_handler(chunk)
            chunks.task_done()

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()

    for i in range(1, size, chunk_size):
        chunk = array[i:i + chunk_size].ndarray()
        chunks.put(chunk)

    chunks.join()


def _process_chunks_simple(array, chunk_handler):
    # Simple, single-threaded version for debugging.
    size = array.shape[0]
    chunk_size = 10

    for i in range(1, size, chunk_size):
        chunk = array[i:i + chunk_size].ndarray()
        chunk_handler(chunk)


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
        elif isinstance(key, (tuple, np.ndarray)):
            sliced_shape.append(len(key))
        else:
            sliced_shape.append(size)
    sliced_shape = tuple(sliced_shape)
    return sliced_shape
