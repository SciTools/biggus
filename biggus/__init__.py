# (C) British Crown Copyright 2012 - 2015, Met Office
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

For example::

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
from __future__ import division

from abc import ABCMeta, abstractproperty, abstractmethod
import __builtin__
import collections
import functools
import itertools
import threading
import Queue
import sys
import warnings

import numpy as np
import numpy.ma as ma


__version__ = '0.13.0'


_SCALAR_KEY_TYPES = (int, np.integer)
_KEY_TYPES = _SCALAR_KEY_TYPES + (slice, tuple, np.ndarray)


def _is_scalar(key):
    return isinstance(key, (int, np.integer))


class AxisSupportError(StandardError):
    """Raised when the operation is not supported over a given axis/axes."""


class Engine(object):
    """
    Represents a way to evaluate lazy expressions.

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def masked_arrays(self, *arrays):
        """
        Return a list of MaskedArray objects corresponding to the given
        biggus Array objects.

        This can be more efficient (and hence faster) than converting the
        individual arrays one by one.

        """
        pass

    @abstractmethod
    def ndarrays(self, *arrays):
        """
        Return a list of NumPy ndarray objects corresponding to the given
        biggus Array objects.

        This can be more efficient (and hence faster) than converting the
        individual arrays one by one.

        """
        pass


Chunk = collections.namedtuple('Chunk', 'keys data')


QUEUE_FINISHED = None
QUEUE_ABORT = Exception


class Node(object):
    """A node of an expression evaluation graph."""

    __metaclass__ = ABCMeta

    def __init__(self):
        self.output_queues = []

    def abort(self):
        """Send the abort signal to all registered output queues."""
        for queue in self.output_queues:
            queue.put(QUEUE_ABORT)

    def add_output_queue(self, output_queue):
        """
        Register a queue so it will receive the output Chunks from this
        Node.

        """
        self.output_queues.append(output_queue)

    def output(self, chunk):
        """
        Dispatch the given Chunk onto all the registered output queues.

        If the chunk is None, it is silently ignored.

        """
        if chunk is not None:
            for queue in self.output_queues:
                queue.put(chunk)

    @abstractmethod
    def run(self):
        pass

    def thread(self):
        """Start a new daemon thread which executes the `run` method."""
        thread = threading.Thread(target=self.run, name=str(self))
        thread.daemon = True
        thread.start()
        return thread


class ProducerNode(Node):
    """
    A data-source node in an expression evaluation graph.

    A ProducerNode corresponds to an Array which simply contains its
    source data. The relevant Array classes are: `NumpyArrayAdapter`,
    `OrthoArrayAdapater`, `ArrayStack`, and `LinearMosaic`.

    """
    def __init__(self, array, iteration_order, masked):
        assert array.ndim == len(iteration_order)
        self.array = array
        self.iteration_order = iteration_order
        self.masked = masked
        super(ProducerNode, self).__init__()

    def run(self):
        """
        Emit the Chunk instances which cover the underlying Array.

        The Array is divided into chunks with a size limit of
        MAX_CHUNK_SIZE which are emitted into all registered output
        queues.

        """
        try:
            # We always slice up the Array into the same chunks, but
            # the order that we traverse those chunks depends on
            # `self.iteration_order`.
            # We use `numpy.ndindex` to iterate through all the chunks,
            # but since it always iterates over the last dimension first
            # we have to transpose `all_cuts` and `cut_shape` ourselves.
            # Then we have to invert the transposition once we have
            # indentified the relevant slices.
            all_cuts = _all_slices_inner(self.array.shape,
                                         always_slices=True)
            all_cuts = [all_cuts[i] for i in self.iteration_order]
            cut_shape = tuple(len(cuts) for cuts in all_cuts)
            inverse_order = [self.iteration_order.index(i) for
                             i in range(len(self.iteration_order))]
            for cut_indices in np.ndindex(*cut_shape):
                key = tuple(cuts[i] for cuts, i in zip(all_cuts, cut_indices))
                key = tuple(key[i] for i in inverse_order)
                # Now we have the slices that describe the next chunk.
                # For example, key might be equivalent to
                # `[11:12, 0:3, :, :]`.
                # Simply "realise" the data for that region and emit it
                # as a Chunk to all registered output queues.
                if self.masked:
                    data = self.array[key].masked_array()
                else:
                    data = self.array[key].ndarray()
                output_chunk = Chunk(key, data)
                self.output(output_chunk)
        except:
            self.abort()
            raise
        else:
            for queue in self.output_queues:
                queue.put(QUEUE_FINISHED)


class ConsumerNode(Node):
    """
    A computation/result-accumulation node in an expression evaluation
    graph.

    A ConsumerNode corresponds to either: an Array which is computed
    from one or more other Arrays; or a container for the result of an
    expressions, such as an in-memory array or file.

    """

    def __init__(self):
        self.input_queues = []
        super(ConsumerNode, self).__init__()

    def add_input_nodes(self, input_nodes):
        """
        Set the given nodes as inputs for this node.

        Creates a limited-size Queue.Queue for each input node and
        registers each queue as an output of its corresponding node.

        """
        self.input_queues = [Queue.Queue(maxsize=3) for _ in input_nodes]
        for input_node, input_queue in zip(input_nodes, self.input_queues):
            input_node.add_output_queue(input_queue)

    @abstractmethod
    def finalise(self):
        """
        Return any remaining partial results.

        Called once all the input chunks have been processed.

        Returns
        -------
        Chunk or None

        """
        pass

    @abstractmethod
    def process_chunks(self, chunks):
        """Process one chunk from each input node."""
        pass

    def run(self):
        """
        Process the input queues in lock-step, and push any results to
        the registered output queues.

        """
        try:
            while True:
                input_chunks = [input.get() for input in self.input_queues]
                for input in self.input_queues:
                    input.task_done()
                if any(chunk is QUEUE_ABORT for chunk in input_chunks):
                    self.abort()
                    return
                if any(chunk is QUEUE_FINISHED for chunk in input_chunks):
                    break
                self.output(self.process_chunks(input_chunks))
            # Finalise the final chunk (process_chunks does this for all
            # but the last chunk).
            self.output(self.finalise())
        except:
            self.abort()
            raise
        else:
            for queue in self.output_queues:
                queue.put(QUEUE_FINISHED)


class StreamsHandlerNode(ConsumerNode):
    """
    A node in an expression graph corresponding to an Array with a
    `streams_handler` method.

    """
    def __init__(self, array, streams_handler):
        self.array = array
        self.streams_handler = streams_handler
        super(StreamsHandlerNode, self).__init__()

    def finalise(self):
        return self.streams_handler.finalise()

    def input_iteration_order(self, iteration_order):
        return self.streams_handler.input_iteration_order(iteration_order)

    def process_chunks(self, chunks):
        return self.streams_handler.process_chunks(chunks)


class NdarrayNode(ConsumerNode):
    """
    An in-memory result node in an expression evaluation graph.

    An NdarrayNode corresponds to either a numpy ndarray instance or a
    MaskedArray instance.

    """

    def __init__(self, array, masked):
        if masked:
            self.result = np.ma.empty(array.shape, dtype=array.dtype)
        else:
            self.result = np.empty(array.shape, dtype=array.dtype)
        super(NdarrayNode, self).__init__()

    def abort(self):
        self.result = None

    def finalise(self):
        pass

    def process_chunks(self, chunks):
        """
        Store the incoming chunk at the corresponding position in the
        result array.

        """
        chunk, = chunks
        if chunk.keys:
            self.result[chunk.keys] = chunk.data
        else:
            self.result[...] = chunk.data


class AllThreadedEngine(Engine):
    """
    Evaluates lazy expressions by creating a thread for each node in the
    expression graph.

    """
    class Group(object):
        """
        A collection of Array instances which are to be evaluated in
        parallel.

        """

        def __init__(self, arrays, indices):
            """
            Creates a collection of Array instances and their
            corresponding indices into the overall list of results.

            Parameters
            ----------
            arrays : iterable of biggus.Array instances
            indices : iterable of int

            """
            self.arrays = arrays
            self.indices = indices
            self._node_cache = {}

        def __repr__(self):
            return 'Group({}, {})'.format(self.arrays, self.indices)

        def _make_node(self, array, iteration_order, masked):
            cache_key = id(array)
            node = self._node_cache.get(cache_key, None)
            if node is None:
                if hasattr(array, 'streams_handler'):
                    node = StreamsHandlerNode(array,
                                              array.streams_handler(masked))
                    iteration_order = node.input_iteration_order(
                        iteration_order)
                    input_nodes = [self._make_node(input_array,
                                                   iteration_order, masked)
                                   for input_array in array.sources]
                    node.add_input_nodes(input_nodes)
                else:
                    node = ProducerNode(array, iteration_order, masked)
                self._node_cache[cache_key] = node
            return node

        def evaluate(self, masked):
            """
            Convert each of the Array instances in this group into its
            corresponding ndarray/MaskedArray.

            Parameters
            ----------
            masked : bool
                Whether to use ndarray or MaskedArray computations.

            Returns
            -------
            list of ndarray or MaskedArray instances

            """
            # Construct nodes starting from the producers.
            result_nodes = []
            result_threads = []
            for array in self.arrays:
                iteration_order = range(array.ndim)
                node = self._make_node(array, iteration_order, masked)
                result_node = NdarrayNode(array, masked)
                result_node.add_input_nodes([node])
                result_threads.append(result_node.thread())
                result_nodes.append(result_node)

            # Start up all the producer/computation threads.
            for node in self._node_cache.itervalues():
                node.thread()

            # Wait for the result threads to finish.
            for thread in result_threads:
                thread.join()

            results = [node.result for node in result_nodes]
            if any(result is None for result in results):
                raise Exception('error during evaluation')
            return results

    def _groups(self, arrays):
        # XXX Placeholder implementation which assumes everything
        # is compatible and can be evaluated in parallel.
        return [self.Group(arrays, range(len(arrays)))]

    def _evaluate(self, arrays, masked):
        # Figure out which arrays should be evaluated in parallel.
        groups = self._groups(arrays)
        # Compile the results.
        all_results = [None] * len(arrays)
        for group in groups:
            ndarrays = group.evaluate(masked)
            for i, ndarray in zip(group.indices, ndarrays):
                all_results[i] = ndarray
        return all_results

    def masked_arrays(self, *arrays):
        return self._evaluate(arrays, True)

    def ndarrays(self, *arrays):
        return self._evaluate(arrays, False)


engine = AllThreadedEngine()
"""
The current lazy evaluation engine.

Defaults to an instance of :class:`AllThreadedEngine`.

"""


class Array(object):
    """
    A virtual array which can be sliced to create smaller virtual
    arrays, or converted to a NumPy ndarray.

    """
    __metaclass__ = ABCMeta

    __hash__ = None

    #: Indicates to client code that the object supports
    #: "orthogonal indexing", which means that slices that are 1d arrays
    #: or lists slice along each dimension independently. This behavior
    #: is similar to Fortran or Matlab, but different than numpy.
    __orthogonal_indexing__ = True

    def __array__(self, dtype=None):
        result = self.ndarray()
        return np.asarray(result, dtype=dtype)

    def __str__(self):
        fmt = '<Array shape=({}) dtype={!r} size={}>'
        return fmt.format(', '.join(str(items) for items in self.shape),
                          self.dtype,
                          size(self))

    def __repr__(self):
        fmt = '<{} shape=({}) dtype={!r}>'
        return fmt.format(type(self).__name__,
                          ', '.join(str(items) for items in self.shape),
                          self.dtype)

    @property
    def fill_value(self):
        """The value used to fill in masked values where necessary."""
        return np.ma.empty(0, dtype=self.dtype).fill_value

    @property
    def nbytes(self):
        """The total number of bytes required to store the array data."""
        return int(np.product(self.shape) * self.dtype.itemsize)

    @property
    def ndim(self):
        """The number of dimensions in this virtual array."""
        return len(self.shape)

    @abstractproperty
    def dtype(self):
        """The datatype of this virtual array."""

    def astype(self, dtype):
        """Copy of the array, cast to a specified type."""
        return AsDataTypeArray(self, dtype)

    @abstractproperty
    def shape(self):
        """The shape of the virtual array as a tuple."""

    def __getitem__(self, keys):
        """Returns a new Array by slicing using _getitem_full_keys."""
        keys = _full_keys(keys, self.ndim)
        # To prevent numpy complaining about "None in arr" we don't
        # simply do "np.newaxis in keys".
        new_axis_exists = any(key is np.newaxis for key in keys)
        if new_axis_exists:
            # Make a NewAxisArray containing this array and 0 new axes.
            array = NewAxesArray(self, [0] * (self.ndim + 1))
            indexed_array = array[keys]
        else:
            indexed_array = self._getitem_full_keys(keys)
        return indexed_array

    def _getitem_full_keys(self, keys):
        """
        Returns a new Array by slicing this virtual array.

        Parameters
        ----------
        keys - iterable of keys
            The keys to index the array with. The default ``__getitem__``
            removes all ``np.newaxis`` objects, and will be of length
            array.ndim.

        Note: This method must be overridden if ``__getitem__`` is defined by
        :meth:`Array.__getitem__`.

        """
        raise NotImplementedError('_getitem_full_keys should be overridden.')

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

    def _normalise_keys(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        # This weird check is safe against keys[-1] being an ndarray.
        if keys and isinstance(keys[-1], type(Ellipsis)):
            keys = keys[:-1]
        if len(keys) > self.ndim:
            raise IndexError('too many keys')
        for key in keys:
            if not(isinstance(key, _KEY_TYPES)):
                raise TypeError('invalid index: {!r}'.format(key))
        return keys

    def transpose(self, axis=None):
        """
        Permute the dimensions of the array.

        Parameters
        ----------
        axes - list of ints, optional
            By default, reverse the dimensions, otherwise permute the axes
            according to the values given.

        """
        return TransposedArray(self, axis)

    def __add__(self, other):
        try:
            return add(self, other)
        except TypeError:
            return NotImplemented

    def __radd__(self, other):
        try:
            return add(other, self)
        except TypeError:
            return NotImplemented

    def __sub__(self, other):
        try:
            return sub(self, other)
        except TypeError:
            return NotImplemented

    def __rsub__(self, other):
        try:
            return sub(other, self)
        except TypeError:
            return NotImplemented

    def __mul__(self, other):
        try:
            return multiply(self, other)
        except TypeError:
            return NotImplemented

    def __rmul__(self, other):
        try:
            return multiply(other, self)
        except TypeError:
            return NotImplemented

    def __floordiv__(self, other):
        try:
            return floor_divide(self, other)
        except TypeError:
            return NotImplemented

    def __rfloordiv__(self, other):
        try:
            return floor_divide(other, self)
        except TypeError:
            return NotImplemented

    def __div__(self, other):
        try:
            return divide(self, other)
        except TypeError:
            return NotImplemented

    def __rdiv__(self, other):
        try:
            return divide(other, self)
        except TypeError:
            return NotImplemented

    def __truediv__(self, other):
        try:
            return true_divide(self, other)
        except TypeError:
            return NotImplemented

    def __rtruediv__(self, other):
        try:
            return true_divide(other, self)
        except TypeError:
            return NotImplemented

    def __pow__(self, other):
        # n.b. __builtin__.pow() allows a modulus. That interface is not
        # supported here as it isn't clear what the benefit is at this stage.
        try:
            return power(self, other)
        except TypeError:
            return NotImplemented

    def __rpow__(self, other):
        try:
            return power(other, self)
        except TypeError:
            return NotImplemented

#    def __mod__(self, other):
#        try:
#            return mod(self, other)
#        except TypeError:
#            return NotImplemented
#
#    def __rmod__(self, other):
#        try:
#            return mod(self, other)
#        except TypeError:
#            return NotImplemented


class ArrayContainer(Array):
    "A biggus.Array which passes calls through to the contained array."
    def __init__(self, contained_array):
        self.array = contained_array

    def __repr__(self):
        return 'ArrayContainer({!r})'.format(self.array)

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, keys):
        # Pass indexing to the contained array. For ArrayContainer types
        # which implement their own complex __getitem__ behaviour,
        # overriding this and _getitem_full_keys may be necessary. See also
        # BroadcastArray and TransposedArray.
        return self.array.__getitem__(keys)

    def ndarray(self):
        try:
            return self.array.ndarray()
        except AttributeError:
            return np.array(self.array)

    def masked_array(self):
        try:
            return self.array.masked_array()
        except AttributeError:
            return np.ma.masked_array(self.array)


class NewAxesArray(ArrayContainer):
    def __init__(self, array, new_axes):
        """
        Creates an array which has new axes (i.e. length 1) at the
        specified locations.

        Parameters
        ----------
        array - array like
            The array upon which to put new axes
        new_axes - iterable of length array.ndim + 1
            The number of new axes for each axis.
            e.g. [2, 1, 0] for a 2d array gain two new axes on the left hand
            side, one in the middle, and 0 on the right hand side.

        """
        super(NewAxesArray, self).__init__(array)

        if array.ndim + 1 != len(new_axes):
            raise ValueError('The new_axes must have length {} but was '
                             'actually length {}.'.format(array.ndim + 1,
                                                          len(new_axes)))
        new_axes = np.array(new_axes)
        dtype_kind = new_axes.dtype.type
        if (not issubclass(dtype_kind, np.integer) or np.any(new_axes < 0)):
            raise ValueError('Only positive integer types may be used for '
                             'new_axes.')

        self._new_axes = new_axes

    @property
    def ndim(self):
        return np.sum(self._new_axes) + self.array.ndim

    @property
    def shape(self):
        shape = list(self.array.shape)
        # Starting from the higher dimensions, insert 1s at the locations
        # of new axes.
        for axes, n_new_axes in reversed(list(enumerate(self._new_axes))):
            for _ in range(n_new_axes):
                shape.insert(axes, 1)
        return tuple(shape)

    def _newaxis_keys(self):
        # Compute the keys needed to produce an array of appropriate newaxis.
        keys = [slice(None)] * self.array.ndim
        # Starting from the higher dimensions, insert np.newaxis at the
        # locations of new axes.
        for axes, n_new_axes in reversed(list(enumerate(self._new_axes))):
            for _ in range(n_new_axes):
                keys.insert(axes, np.newaxis)
        return tuple(keys)

    def _is_newaxis(self):
        is_newaxis = [False] * self.array.ndim
        for axes, n_new_axes in reversed(list(enumerate(self._new_axes))):
            for _ in range(n_new_axes):
                is_newaxis.insert(axes, True)
        return tuple(is_newaxis)

    def __getitem__(self, keys):
        # We don't want to implement _getitem_full_keys here, as we
        # don't want a potentially deep nesting of NewAxesArrays. Instead
        # we work out where newaxis objects are in the existing array, and
        # add any new ones that are requested.

        keys = _full_keys(keys, self.ndim)
        new_axes = self._new_axes

        # Strip out an deal with any keys which are for new axes.
        new_axes = new_axes.copy()
        axes_to_combine = []
        is_newaxis = list(self._is_newaxis())
        contained_array_keys = []
        existing_array_axis = 0
        broadcast_dict = {}

        for key_index, key in enumerate(keys):
            if key is np.newaxis:
                new_axes[existing_array_axis] += 1
                continue

            if is_newaxis.pop(0):
                # We're indexing a new_axes axes.
                if _is_scalar(key):
                    if -1 <= key < 1:
                        new_axes[existing_array_axis] -= 1
                    else:
                        raise IndexError('index {} is out of bounds for axis '
                                         '{} with size 1'.format(key,
                                                                 key_index))
                elif isinstance(key, slice):
                    new_size = len(range(*key.indices(1)))
                    if new_size != 1:
                        broadcast_dict[key_index] = new_size
                elif isinstance(key, tuple):
                    for index in key:
                        if not -1 <= index < 1:
                            raise IndexError('index {} is out of bounds for '
                                             'axis {} with size 1'
                                             ''.format(key, key_index))
                    broadcast_dict[key_index] = len(key)
                else:
                    raise NotImplementedError('NewAxesArray indexing not yet '
                                              'supported for {} keys.'
                                              ''.format(type(key).__name__))
            else:
                # We're indexing a dimension of self.array.
                if _is_scalar(key):
                    # One of the dimensions of the existing data is to be
                    # removed, so we can combine the new_axes to the left
                    # and right of this axes into a single value.
                    axes_to_combine.append(existing_array_axis)
                contained_array_keys.append(key)
                existing_array_axis += 1

        new_axes = list(new_axes)
        for axis in sorted(axes_to_combine, reverse=True):
            new_axes[axis] += new_axes.pop(axis + 1)
        new_array = NewAxesArray(self.array[tuple(contained_array_keys)],
                                 new_axes)
        if broadcast_dict:
            new_array = BroadcastArray(new_array, broadcast_dict)
        return new_array

    def ndarray(self):
        array = super(NewAxesArray, self).ndarray()
        return array.__getitem__(self._newaxis_keys())

    def masked_array(self):
        array = super(NewAxesArray, self).masked_array()
        return array.__getitem__(self._newaxis_keys())


class BroadcastArray(ArrayContainer):
    def __init__(self, array, broadcast, leading_shape=()):
        """
        Parameters
        ----------
        array : array like
            The array to broadcast. Only length 1 dimensions, or those
            already being broadcast, may be (further) broadcast.
        broadcast : dict
            A mapping of broadcast axis to broadcast length.
        leading_shape : iterable
            A shape to put on the leading dimension of the array.

        >>> array = BroadcastArray(np.empty([1, 4]),
        ...                        broadcast={0: 10},
        ...                        leading_shape=(5,))
        >>> array.shape
        (5, 10, 4)

        """
        # To avoid nesting broadcast arrays within broadcast arrays, we
        # simply copy the existing broadcast, and apply this broadcast on
        # top of it (in a new BroadcastArray instance).
        if isinstance(array, BroadcastArray):
            new_broadcast_dict = array._broadcast_dict.copy()
            leading_shape = tuple(leading_shape) + array._leading_shape
            array = array.array
            new_broadcast_dict.update(broadcast)
            broadcast = new_broadcast_dict

        super(BroadcastArray, self).__init__(array)
        self._broadcast_dict = broadcast

        # Compute the broadcast shape.
        shape = self._shape_from_broadcast_dict(self.array.shape, broadcast)
        for length in leading_shape:
            if length < 1:
                raise ValueError('Leading shape must all be >=1.')
        self._leading_shape = tuple(leading_shape)
        self._broadcast_shape = tuple(shape)
        self._shape = self._leading_shape + self._broadcast_shape

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, keys):
        # Inherit the behaviour from Array, **not** from ArrayContainer.
        return super(ArrayContainer, self).__getitem__(keys)

    def _getitem_full_keys(self, keys):
        array_keys = []
        new_broadcast_dict = {}
        axis_offset = 0

        # Take off the leading shape, and use the sliced_shape functionality
        # to compute the new leading shape size.
        leading_shape = self._leading_shape
        leading_shape_len = len(leading_shape)
        leading_shape = _sliced_shape(leading_shape, keys[:leading_shape_len])

        axis_offset -= leading_shape_len

        n_axes_removed = 0

        for axis, key in enumerate(keys[leading_shape_len:],
                                   start=leading_shape_len):
            concrete_axis = axis + axis_offset
            if concrete_axis not in self._broadcast_dict:
                if _is_scalar(key):
                    n_axes_removed += 1
                array_keys.append(key)
            else:
                existing_size = self._shape[axis]
                if isinstance(key, slice):
                    # We just want to preserve the dimension. We will deal
                    # with the broadcasting of the length.
                    array_keys.append(slice(None))

                    # TODO: Compute this without creating a range object.
                    size = len(range(*key.indices(existing_size)))
                    new_broadcast_dict[concrete_axis - n_axes_removed] = size
                elif _is_scalar(key):
                    if not -existing_size <= key < existing_size:
                        raise IndexError('index {} is out of bounds for axis '
                                         '{} with size 1'.format(key, axis))
                    else:
                        # We want to index the broadcast dimension.
                        array_keys.append(0)
                        n_axes_removed += 1
                else:
                    raise NotImplementedError('Indexing with type {} not yet '
                                              'implemented.'
                                              ''.format(type(key)))
        # Try to avoid a copy of self.array if we can.
        if all(key == slice(None) for key in array_keys):
            sub_array = self.array
        else:
            sub_array = self.array[tuple(array_keys)]
        return type(self)(sub_array, new_broadcast_dict, leading_shape)

    @classmethod
    def broadcast_arrays(cls, array1, array2):
        """
        Broadcast two arrays against each other.

        Returns
        -------
        broadcast_array1 : array1 or a broadcast of array1
        broadcast_array2 : array2 or a broadcast of array2
            The returned arrays will be broadcast against each other.
            Any array which is already in full broadcast shape will be
            returned unchanged.

        """
        shape, bcast_kwargs1, bcast_kwargs2 = (
            cls._compute_broadcast_kwargs(array1.shape, array2.shape))
        if any(bcast_kwargs1.values()):
            array1 = cls(array1, **bcast_kwargs1)

        if any(bcast_kwargs2.values()):
            array2 = cls(array2, **bcast_kwargs2)

        return array1, array2

    @staticmethod
    def _compute_broadcast_shape(shape1, shape2):
        """
        Given two shapes, use numpy's broadcasting rules to compute the
        broadcasted shape.

        """
        # Rule 1: If the two arrays differ in their number of dimensions, the
        # shape of the array with fewer dimensions is padded with ones on its
        # leading (left) side.
        s1, s2 = list(shape1), list(shape2)
        len_diff = len(s1) - len(s2)
        if len_diff > 0:
            s2[0:0] = [1] * len_diff
        else:
            s1[0:0] = [1] * -len_diff

        # Rule 2: If the shape of the two arrays does not match in any
        # dimension, the array with shape equal to 1 in that dimension is
        # stretched to match the other shape.
        shape = []
        for size1, size2 in zip(s1, s2):
            if size1 == size2:
                shape.append(size1)
            elif size1 == 1:
                shape.append(size2)
            elif size2 == 1:
                shape.append(size1)
            else:
                # Rule 3: If in any dimension the sizes disagree and neither is
                # equal to 1, an error is raised.
                raise ValueError('operands could not be broadcast together '
                                 'with shapes ({}) ({})'
                                 ''.format(','.join(map(str, shape1)),
                                           ','.join(map(str, shape2))))
        return tuple(shape)

    @classmethod
    def _compute_broadcast_kwargs(cls, shape1, shape2):
        """
        Given two shapes, compute the broadcast shape, along with the keywords
        needed to produce BroadcastArrays with arrays of given shape.

        Parameters
        ----------
        shape1 : iterable
        shape2 : iterable
            The two shapes to broadcast against one another.

        Returns
        -------
        full_shape : iterable
            The full broadcast shape.
        broadcast_kwargs1 : dict
        broadcast_kwargs2 : dict
            Keywords which are suitably passed through to a BroadcastArray
            to take the original array, and broadcast to the full broadcast
            shape.

        """
        full_shape = cls._compute_broadcast_shape(shape1, shape2)

        bcast_kwargs1 = {'broadcast': {}, 'leading_shape': ()}
        bcast_kwargs2 = {'broadcast': {}, 'leading_shape': ()}

        ndim_diff = len(shape1) - len(shape2)

        s1_offset = s2_offset = 0

        if ndim_diff > 0:
            s2_offset = ndim_diff
            bcast_kwargs2['leading_shape'] = full_shape[:s2_offset]
        elif len(shape1) < len(shape2):
            s1_offset = abs(ndim_diff)
            bcast_kwargs1['leading_shape'] = full_shape[:s1_offset]

        for ax, (full, s1) in enumerate(zip(full_shape[s1_offset:], shape1)):
            if full != s1:
                bcast_kwargs1['broadcast'][ax] = full

        for ax, (full, s2) in enumerate(zip(full_shape[s2_offset:], shape2)):
            if full != s2:
                bcast_kwargs2['broadcast'][ax] = full
        return full_shape, bcast_kwargs1, bcast_kwargs2

    @classmethod
    def _shape_from_broadcast_dict(cls, orig_shape, broadcast_dict):
        """Using a broadcast dictionary, compute the broadcast shape."""
        shape = list(orig_shape)
        for axis, length in broadcast_dict.items():
            if not 0 <= axis < len(shape):
                raise ValueError('Axis {} out of range [0, {})'
                                 ''.format(axis, len(shape)))
            if length < 0:
                raise ValueError('Axis length must be positive. Got {}.'
                                 ''.format(length))
            if shape[axis] != 1:
                raise ValueError('Attempted to broadcast axis {} which is of '
                                 'length {}.'.format(axis, shape[axis]))
            shape[axis] = length
        return tuple(shape)

    @classmethod
    def _broadcast_numpy_array(cls, array, broadcast_dict, leading_shape=()):
        """Broadcast a numpy array according to the broadcast_dict."""
        from numpy.lib.stride_tricks import as_strided
        shape = cls._shape_from_broadcast_dict(array.shape, broadcast_dict)
        shape = tuple(leading_shape) + shape
        strides = [0] * len(leading_shape) + list(array.strides)
        for broadcast_axis in broadcast_dict:
            strides[broadcast_axis + len(leading_shape)] = 0
        return as_strided(array, shape=tuple(shape), strides=tuple(strides))

    def ndarray(self):
        array = super(BroadcastArray, self).ndarray()
        return self._broadcast_numpy_array(array, self._broadcast_dict,
                                           self._leading_shape)

    def masked_array(self):
        ma = super(BroadcastArray, self).masked_array()
        mask = ma.mask
        array = self._broadcast_numpy_array(ma.data, self._broadcast_dict,
                                            self._leading_shape)
        if isinstance(mask, np.ndarray):
            mask = self._broadcast_numpy_array(mask, self._broadcast_dict,
                                               self._leading_shape)
        return np.ma.masked_array(array, mask=mask)


class AsDataTypeArray(ArrayContainer):
    def __init__(self, array, dtype):
        """
        Cast the given array to the specified dtype.

        Parameters
        ----------
        array : array like
            The array to cast to ``dtype``.
        dtype : valid numpy.dtype argument
            The dtype to cast the data to. This will be
            passed through to :func:`numpy.dtype`.

        """
        super(AsDataTypeArray, self).__init__(array)
        self._dtype = np.dtype(dtype)

    @property
    def dtype(self):
        return self._dtype

    def astype(self, dtype):
        return type(self)(self.array, dtype)

    def __getitem__(self, keys):
        # Apply the indexing to the contained array, then instantly
        # re-apply the astype.
        return type(self)(super(AsDataTypeArray, self).__getitem__(keys),
                          self.dtype)

    def ndarray(self):
        return super(AsDataTypeArray,
                     self).ndarray().astype(self.dtype)

    def masked_array(self):
        return super(AsDataTypeArray,
                     self).masked_array().astype(self.dtype)


class ConstantArray(Array):
    """
    An Array which is completely filled with a single value.

    Parameters
    ----------
    shape : int or sequence of ints
        The shape for the new Array.
    value : obj, optional
        The value to fill the Array with. Defaults to 0.0.
    dtype : obj, optional
        Object to be converted to data type. Default is None which
        instructs the data type to be determined as the minimum required
        to hold the given value.

    Returns
    -------
    Array
        An Array entirely filled with 'value'.

    """
    def __init__(self, shape, value=0.0, dtype=None):
        if isinstance(shape, basestring):
            shape = (shape,)
        else:
            try:
                shape = tuple(shape)
            except TypeError:
                shape = (shape,)
        self._shape = tuple(map(int, shape))
        data = np.array([value], dtype=dtype)
        self.value = data[0]
        self._dtype = data.dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, keys):
        # newaxis is handled within _sliced_shape, so we override __getitem__,
        # not _getitem_full_keys.
        shape = _sliced_shape(self.shape, keys)
        return ConstantArray(shape, self.value, self._dtype)

    def ndarray(self):
        result = np.empty(self.shape, self._dtype)
        result.fill(self.value)
        return result

    def masked_array(self):
        result = np.ma.empty(self.shape, self._dtype)
        result.fill(self.value)
        return result


def zeros(shape, dtype=float):
    """
    Return an Array which is completely filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        The shape for the new Array.
    dtype : obj, optional
        Object to be converted to data type. Default is `float`.

    Returns
    -------
    Array
        An Array entirely filled with zeros.

    """
    return ConstantArray(shape, dtype=dtype)


def ones(shape, dtype=float):
    """
    Return an Array which is completely filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        The shape for the new Array.
    dtype : obj, optional
        Object to be converted to data type. Default is `float`.

    Returns
    -------
    Array
        An Array entirely filled with ones.

    """
    return ConstantArray(shape, 1, dtype=dtype)


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
        return _sliced_shape(self.concrete.shape, self._keys)

    def _cleanup_new_key(self, key, size, axis):
        """
        Return a key of type int, slice, or tuple that is guaranteed
        to be valid for the given dimension size.

        Raises IndexError/TypeError for invalid keys.

        """
        if _is_scalar(key):
            if key >= size or key < -size:
                msg = 'index {0} is out of bounds for axis {1} with' \
                      ' size {2}'.format(key, axis, size)
                raise IndexError(msg)
        elif isinstance(key, slice):
            pass
        elif isinstance(key, np.ndarray) and key.dtype == np.dtype('bool'):
            if key.size > size:
                msg = 'too many boolean indices. Boolean index array ' \
                      'of size {0} is greater than axis {1} with ' \
                      'size {2}'.format(key.size, axis, size)
                raise IndexError(msg)
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
        if _is_scalar(new_key):
            if new_key >= size or new_key < -size:
                msg = 'index {0} is out of bounds for axis {1}' \
                      ' with size {2}'.format(new_key, axis, size)
                raise IndexError(msg)
            result_key = indices[new_key]
        elif isinstance(new_key, slice):
            result_key = indices.__getitem__(new_key)
        elif isinstance(new_key, np.ndarray) and \
                new_key.dtype == np.dtype('bool'):
            # Numpy boolean indexing.
            if new_key.size > size:
                msg = 'too many boolean indices. Boolean index array ' \
                      'of size {0} is greater than axis {1} with ' \
                      'size {2}'.format(new_key.size, axis, size)
                raise IndexError(msg)
            result_key = tuple(np.array(indices)[new_key])
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

    def _getitem_full_keys(self, keys):
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
            if _is_scalar(src_key):
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
                is_scalar = map(_is_scalar, cut_keys)
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

        >>> ortho = OrthoArrayAdapter(ConstantArray(shape=[100, 200, 300]))
        >>> ortho.shape
        (100, 200, 300)
        >>> ortho[(0, 3, 4), :, (1, 9)].shape
        (3, 200, 2)

    A netCDF4.Variable instance is an example orthogonal concrete array.

    NB. Orthogonal indexing contrasts with NumPy "fancy indexing" where
    multiple iterable index keys are zipped together to allow the
    selection of sparse locations.

    """
    def _apply_keys(self):
        array = self.concrete.__getitem__(self._keys)
        return array


def _pairwise(iterable):
    """
    itertools recipe
    "s -> (s0,s1), (s1,s2), (s2, s3), ...

    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def _groups_of(length, total_length):
    """
    Return an iterator of tuples for slicing, in 'length' chunks.

    Parameters
    ----------
    length : int
        Length of each chunk.
    total_length : int
        Length of the object we are slicing

    Returns
    -------
    iterable of tuples
        Values defining a slice range resulting in length 'length'.

    """
    indices = tuple(range(0, total_length, length)) + (None, )
    return _pairwise(indices)


class TransposedArray(ArrayContainer):
    def __init__(self, array, axes=None):
        """
        Permute the dimensions of an array.

        Parameters
        ----------
        array - array like
            The array to transpose
        axes - list of ints, optional
            By default, reverse the dimensions, otherwise permute the axes
            according to the values given.

        """
        super(TransposedArray, self).__init__(array)
        if axes is None:
            axes = np.arange(array.ndim)[::-1]
        elif len(axes) != array.ndim:
            raise ValueError('Incorrect number of dimensions.')
        self.axes = axes
        self._forward_axes_map = {src: dest for dest, src in enumerate(axes)}
        self._inverse_axes_map = {dest: src for dest, src in enumerate(axes)}

    def __repr__(self):
        return 'TransposedArray({!r}, {!r})'.format(self.array, self.axes)

    def _apply_axes_mapping(self, target, inverse=False):
        """
        Apply the transposition to the target iterable.

        Parameters
        ----------
        target - iterable
            The iterable to transpose. This would be suitable for things
            such as a shape as well as a list of ``__getitem__`` keys.
        inverse - bool
            Whether to map old dimension to new dimension (forward), or
            new dimension to old dimension (inverse). Default is False
            (forward).

        Returns
        -------
        A tuple derived from target which has been ordered based on the new
        axes.

        """
        if len(target) != self.ndim:
            raise ValueError('The target iterable is of length {}, but '
                             'should be of length {}.'.format(len(target),
                                                              self.ndim))
        if inverse:
            axis_map = self._inverse_axes_map
        else:
            axis_map = self._forward_axes_map

        result = [None] * self.ndim
        for axis, item in enumerate(target):
            result[axis_map[axis]] = item
        return tuple(result)

    @property
    def shape(self):
        return self._apply_axes_mapping(self.array.shape)

    @property
    def ndim(self):
        return self.array.ndim

    def __getitem__(self, keys):
        # Inherit the behaviour from Array, **not** from ArrayContainer,
        # thus meaning we must implement _getitem_full_keys.
        return super(ArrayContainer, self).__getitem__(keys)

    def _getitem_full_keys(self, keys):
        new_transpose_order = list(self.axes)

        # Map the keys in transposed space back to pre-transposed space.
        remapped_keys = list(self._apply_axes_mapping(keys, inverse=True))

        # Apply the keys to the pre-transposed array.
        new_arr = self.array[tuple(remapped_keys)]

        # Compute the new scalar axes in terms of old (pre-transpose)
        # dimension numbers.
        new_scalar_axes = [dim for dim, key in enumerate(remapped_keys)
                           if _is_scalar(key)]

        # Compute the new transpose axes by successively taking the highest
        # new scalar axes, and removing it from the axes mapping. We must
        # remember that any axis greater than the removed dimension must
        # also be reduced by 1.
        while new_scalar_axes:
            # Take the highest scalar axis.
            scalar_axis = new_scalar_axes.pop()
            new_transpose_order = [axis - 1 if axis >= scalar_axis else axis
                                   for axis in new_transpose_order
                                   if axis != scalar_axis]

        return TransposedArray(new_arr, new_transpose_order)

    def ndarray(self):
        array = super(TransposedArray, self).ndarray()
        return array.transpose(self.axes)

    def masked_array(self):
        array = super(TransposedArray, self).masked_array()
        return array.transpose(self.axes)


class ArrayStack(Array):
    """
    An Array made from a homogeneous array of other Arrays.

    Parameters
    ----------
    stack : array-like
        The array of Arrays to be stacked, where each Array must be of
        the same shape.

    """
    def __init__(self, stack):
        stack = np.require(stack, dtype='O')
        first_array = stack.flat[0]
        item_shape = first_array.shape
        dtype = first_array.dtype
        fill_value = first_array.fill_value

        if np.issubdtype(dtype, np.floating):
            def fill_value_ok(array):
                return array.fill_value == fill_value or (
                    np.isnan(fill_value) and np.isnan(array.fill_value))
        else:
            def fill_value_ok(array):
                return array.fill_value == fill_value

        for array in stack.flat:
            if not isinstance(array, Array):
                raise ValueError('sub-array must be subclass of Array')
            if fill_value is not None and not fill_value_ok(array):
                fill_value = None
            ok = array.shape == item_shape and array.dtype == dtype
            if not ok:
                raise ValueError('invalid sub-array')

        self._stack = stack
        self._item_shape = item_shape
        self._dtype = dtype
        if fill_value is None:
            self._fill_value = np.ma.empty(0, dtype=dtype).fill_value
        else:
            self._fill_value = fill_value

    def __deepcopy__(self, memo):
        # We override deepcopy here as a result of
        # https://github.com/SciTools/biggus/issues/157.
        from copy import deepcopy

        if np.isfortran(self._stack):
            from functools import partial

            deepcopy_with_memo = partial(deepcopy, memo=memo)
            deepcopy_elementwise = np.vectorize(deepcopy_with_memo,
                                                otypes=[np.object])
            result = deepcopy_elementwise(self._stack)
            memo[id(self._stack)] = result

        # Implement a standard deepcopy. We've dealt with any issues already.
        cls = self.__class__
        self_copied = cls.__new__(cls)
        self_copied.__dict__.update(deepcopy(self.__dict__, memo))
        return self_copied

    @property
    def dtype(self):
        return self._dtype

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def shape(self):
        return self._stack.shape + self._item_shape

    def _getitem_full_keys(self, keys):
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
            assert _is_scalar(key)
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

    @staticmethod
    def multidim_array_stack(arrays, shape, order='C'):
        """
        Create an N-dimensional ArrayStack from the sequence of Arrays of the
        same shape.

        Example usage: stacking 6 Arrays, each of shape (768, 1024) to a
        specified shape (2, 3) will result in an ArrayStack of shape
        (2, 3, 768, 1024).

        Parameters
        ----------
        arrays : sequence of Arrays
            The sequence of Arrays to be stacked, where each Array must be of
            the same shape.
        shape : sequence of ints
            Shape of the stack, (2, 3) in the above example.
        order : {'C', 'F'}, optional
            Use C (C) or FORTRAN (F) index ordering.

        Returns
        -------
        ArrayStack
            With shape corresponding to tuple(stack shape) + tuple(Array.shape)
            where each Array in the stack must be of the same shape.

        """
        arrays = np.require(arrays, dtype='O')

        order = order.lower()

        # Ensure a suitable shape has been specified.
        size_msg = "total size of new array must be unchanged"
        try:
            if np.product(shape) != np.product(arrays.shape):
                raise ValueError(size_msg)
            if arrays.ndim > 1:
                raise ValueError("multidimensional stacks not yet supported")
        except AttributeError:
            if np.product(shape) != len(arrays):
                raise ValueError(size_msg)
        # Hold the subdivided array
        subdivided_array = arrays

        # Recursively subdivide to create an ArrayStack with specified shape.
        for length in shape[::-1]:
            # Hold the array before this iterations subdivide.
            array_before_subdivide = subdivided_array
            subdivided_array = []

            if order == 'f':
                num = len(array_before_subdivide) // length
                if length == len(array_before_subdivide):
                    slc = [slice(None)]
                else:
                    slc = [slice(i, None, num) for i in range(num)]
                for sc in slc:
                    sub = ArrayStack(np.array(array_before_subdivide[sc],
                                              dtype=object))
                    subdivided_array.append(sub)
            elif order == 'c':
                for start, stop in _groups_of(length,
                                              len(array_before_subdivide)):
                    sub = ArrayStack(np.array(
                        array_before_subdivide[start:stop], dtype=object))
                    subdivided_array.append(sub)
            else:
                raise TypeError('order not understood')
        else:
            # Last iteration, length of the array will be equal to 1.
            return subdivided_array[0]


class LinearMosaic(Array):
    def __init__(self, tiles, axis):
        if not isinstance(tiles, collections.Iterable):
            tiles = [tiles]
        tiles = np.array(tiles, dtype='O', ndmin=1)
        if tiles.ndim != 1:
            raise ValueError('the tiles array must be 1-dimensional')
        first = tiles[0]
        if not isinstance(first, Array):
            raise ValueError('sub-array must be subclass of Array')
        if not(0 <= axis < first.ndim):
            msg = 'invalid axis for {0}-dimensional tiles'.format(first.ndim)
            raise ValueError(msg)
        # Make sure all the tiles are compatible
        common_shape = list(first.shape)
        common_dtype = first.dtype
        common_fill_value = first.fill_value
        if np.issubdtype(common_dtype, np.floating):
            def fill_value_ok(array):
                return array.fill_value == common_fill_value or (
                    np.isnan(common_fill_value) and np.isnan(array.fill_value))
        else:
            def fill_value_ok(array):
                return array.fill_value == common_fill_value
        del common_shape[axis]
        for tile in tiles[1:]:
            if not isinstance(tile, Array):
                raise ValueError('sub-array must be subclass of Array')
            shape = list(tile.shape)
            del shape[axis]
            if shape != common_shape:
                raise ValueError('inconsistent tile shapes')
            if tile.dtype != common_dtype:
                raise ValueError('inconsistent tile dtypes')
            if common_fill_value is not None and not fill_value_ok(tile):
                common_fill_value = None

        self._tiles = tiles
        self._axis = axis
        self._cached_shape = None
        if common_fill_value is None:
            self._fill_value = np.ma.empty(0, dtype=common_dtype).fill_value
        else:
            self._fill_value = common_fill_value

    @property
    def dtype(self):
        return self._tiles[0].dtype

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def shape(self):
        if self._cached_shape is None:
            shape = list(self._tiles[0].shape)
            for tile in self._tiles[1:]:
                shape[self._axis] += tile.shape[self._axis]
            self._cached_shape = tuple(shape)
        return self._cached_shape

    def _getitem_full_keys(self, full_keys):
        # Starting backwards, include all keys once a ``key != slice(None)``
        # has been found. This will give us only the keys that are really
        # necessary and thus allow us a shortcut through the indexing.
        keys = []
        non_full_slice_found = False
        for key in full_keys[::-1]:
            if isinstance(key, np.ndarray) or key != slice(None):
                non_full_slice_found = True
            if non_full_slice_found:
                keys.append(key)
        keys = tuple(keys[::-1])

        axis = self._axis
        if len(keys) <= axis:
            # If there aren't enough keys to affect the tiling axis
            # then it's safe to just pass the keys to each tile.
            tile = self._tiles[0]
            tiles = [tile[keys] for tile in self._tiles]
            scalar_keys = filter(_is_scalar, keys)
            result = LinearMosaic(tiles, axis - len(scalar_keys))
        else:
            axis_lengths = [tile.shape[axis] for tile in self._tiles]
            offsets = np.cumsum([0] + axis_lengths[:-1])
            splits = offsets - 1
            axis_key = keys[axis]
            if _is_scalar(axis_key):
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
                    axis_indices = list(zip(*group_of_pairs))[0]
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
                        tile_slice[axis] = slice(start, stop, step)
                    else:
                        tile_slice[axis] = axis_indices

                    tiles.append(tile[tuple(tile_slice)])
                if isinstance(axis_key, slice) and \
                        axis_key.step is not None and axis_key.step < 0:
                    tiles.reverse()
                # Adjust the axis of the new mosaic to account for any
                # scalar keys prior to our current mosaic axis.
                new_axis = axis
                for key in keys[:axis]:
                    if _is_scalar(key):
                        new_axis -= 1
                result = LinearMosaic(tiles, new_axis)
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
    return engine.ndarrays(*arrays)


def masked_arrays(arrays):
    """
    Return a list of NumPy masked array objects corresponding to the given
    biggus Array objects.

    This can be more efficient (and hence faster) than converting the
    individual arrays one by one.

    """
    return engine.masked_arrays(*arrays)


#: The maximum number of bytes per chunk to allow when processing an array in
#: "bite-size" chunks. Chunks are determined by the _all_slices function, where
#: an assumption is made that data type nbytes is 8.
#: The value has been empirically determined to
#: provide vaguely near optimal performance under certain conditions.
MAX_CHUNK_SIZE = 8 * 1024 * 1024 * 2


def _all_slices(array):
    return _all_slices_inner(array.shape)


def _all_slices_inner(shape, always_slices=False):
    # Return the slices for each dimension which ensure complete
    # coverage by chunks no larger than MAX_CHUNK_SIZE.
    # e.g. For a float32 array of shape (100, 768, 1024) the slices are:
    #   (0, 1, 2, ..., 99),
    #   (slice(0, 256), slice(256, 512), slice(512, 768)),
    #   (slice(None)

    # Fix the item size to a single element. `n_elems` will be updated as we
    # traverse the dimensions of shape so that it equals the number of bytes
    # that one item in the current dimension represents.
    n_elems = 1
    all_slices = []
    # We walk through the dimensions, starting from the RHS,
    # and keep track of the total size of one item in the current dimension
    # in the n_elems variable.
    for i, size in reversed(list(enumerate(shape))):
        # Check to see if the whole of this dimension can fit into a single
        # chunk.
        if size * n_elems <= MAX_CHUNK_SIZE:
            slices = (slice(None),)
        # Otherwise, determine if previous dimensions have already saturated
        # MAX_CHUNK_SIZE. If so, we need to pick off each item from this
        # dimension.
        elif n_elems > MAX_CHUNK_SIZE:
            if always_slices:
                slices = [slice(i, i + 1) for i in range(size)]
            else:
                slices = range(size)
        # Otherwise we have found the dimension that reaches the MAX_CHUNK_SIZE
        # limit, so we apply a range which gives chunk sizes as close to the
        # MAX_CHUNK_SIZE as possible.
        else:
            step = MAX_CHUNK_SIZE // n_elems
            slices = []
            for start in range(0, size, step):
                slices.append(slice(start, np.min([start + step, size])))
        n_elems *= size
        all_slices.insert(0, slices)
    return all_slices


def save(sources, targets, masked=False):
    """
    Save the numeric results of each source into its corresponding target.

    Parameters
    ----------
    sources: list
        The list of source arrays for saving from; limited to length 1.
    targets: list
        The list of target arrays for saving to; limited to length 1.
    masked: boolean
        Uses a masked array from sources if True.

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
        if masked:
            target[keys] = array[keys].masked_array()
        else:
            target[keys] = array[keys].ndarray()


class _StreamsHandler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def finalise(self):
        """
        Once a chunk has been processed, this method will be called to
        complete any remaining computation and return a "result chunk"
        (which itself could very well go on for further processing).

        """
        pass

    @abstractmethod
    def input_iteration_order(self, iteration_order):
        pass

    @abstractmethod
    def process_chunks(self, chunks):
        pass


class _AggregationStreamsHandler(_StreamsHandler):
    def __init__(self, array, axis):
        self.array = array
        self.axis = axis
        self.current_keys = None

    @abstractmethod
    def bootstrap(self, processed_chunk_shape):
        """
        Initialise the processing of the next chunk.

        Parameters
        ----------
        processed_chunk_shape : list
            The shape that the current chunk will have once it has
            been computed. For example, for an aggregation of a chunk of
            shape ``(x, y, z)``, over axis 1, ``the processed_chunk_shape``
            would be ``[x, z]``.

        """
        pass

    def input_iteration_order(self, iteration_order):
        order = [i if i < self.axis else i + 1 for i in iteration_order]
        order.append(self.axis)
        return order

    def process_chunks(self, chunks):
        chunk, = chunks
        keys = list(chunk.keys)
        del keys[self.axis]
        result = None
        # If this chunk is a new source of data, do appropriate finalisation
        # of the previous chunk and initialise this one.
        if keys != self.current_keys:
            # If this isn't the first time this method has been called,
            # finalise any data which is waiting to be dealt with.
            if self.current_keys is not None:
                result = self.finalise()

            # Setup the processing of this new chunk.
            shape = list(chunk.data.shape)
            del shape[self.axis]
            self.bootstrap(shape)
            self.current_keys = keys
        self.process_data(chunk.data)
        return result

    @abstractmethod
    def process_data(self, data):
        pass


class _CountStreamsHandler(_AggregationStreamsHandler):
    def bootstrap(self, processed_chunk_shape):
        self.current_shape = processed_chunk_shape
        self.running_count = 0

    def finalise(self):
        count = np.ones(self.current_shape, dtype='i') * self.running_count
        chunk = Chunk(self.current_keys, count)
        return chunk

    def process_data(self, data):
        self.running_count += data.shape[self.axis]


class _CountMaskedStreamsHandler(_AggregationStreamsHandler):
    def bootstrap(self, processed_chunk_shape):
        self.running_count = np.zeros(processed_chunk_shape, dtype='i')

    def finalise(self):
        chunk = Chunk(self.current_keys, self.running_count)
        return chunk

    def process_data(self, data):
        self.running_count += np.ma.count(data, axis=self.axis)


class _MinStreamsHandler(_AggregationStreamsHandler):
    def bootstrap(self, processed_chunk_shape):
        self.result = np.zeros(processed_chunk_shape, dtype=self.array.dtype)

    def finalise(self):
        array = self.result
        # Promote array-scalar to 0-dimensional array.
        if array.ndim == 0:
            array = np.array(array)
        chunk = Chunk(self.current_keys, array)
        return chunk

    def process_data(self, data):
        self.result = np.min(data, axis=self.axis)


class _MinMaskedStreamsHandler(_AggregationStreamsHandler):
    def bootstrap(self, processed_chunk_shape):
        self.result = np.zeros(processed_chunk_shape, dtype=self.array.dtype)

    def finalise(self):
        array = self.result
        # Promote array-scalar to 0-dimensional array.
        if array.ndim == 0:
            array = np.ma.array(array)
        chunk = Chunk(self.current_keys, array)
        return chunk

    def process_data(self, data):
        self.result = np.min(data, axis=self.axis)


class _MaxStreamsHandler(_AggregationStreamsHandler):
    def bootstrap(self, processed_chunk_shape):
        self.result = np.zeros(processed_chunk_shape, dtype=self.array.dtype)

    def finalise(self):
        array = self.result
        # Promote array-scalar to 0-dimensional array.
        if array.ndim == 0:
            array = np.array(array)
        chunk = Chunk(self.current_keys, array)
        return chunk

    def process_data(self, data):
        self.result = np.max(data, axis=self.axis)


class _MaxMaskedStreamsHandler(_AggregationStreamsHandler):
    def bootstrap(self, processed_chunk_shape):
        self.result = np.zeros(processed_chunk_shape, dtype=self.array.dtype)

    def finalise(self):
        array = self.result
        # Promote array-scalar to 0-dimensional array.
        if array.ndim == 0:
            array = np.ma.array(array)
        chunk = Chunk(self.current_keys, array)
        return chunk

    def process_data(self, data):
        self.result = np.max(data, axis=self.axis)


class _SumStreamsHandler(_AggregationStreamsHandler):
    def bootstrap(self, processed_chunk_shape):
        self.running_total = np.zeros(processed_chunk_shape,
                                      dtype=self.array.dtype)

    def finalise(self):
        array = self.running_total
        # Promote array-scalar to 0-dimensional array.
        if array.ndim == 0:
            array = np.array(array)
        chunk = Chunk(self.current_keys, array)
        return chunk

    def process_data(self, data):
        self.running_total += np.sum(data, axis=self.axis)


class _SumMaskedStreamsHandler(_AggregationStreamsHandler):
    def bootstrap(self, processed_chunk_shape):
        self.running_total = np.ma.zeros(processed_chunk_shape,
                                         dtype=self.array.dtype)

    def finalise(self):
        array = self.running_total
        # Promote array-scalar to 0-dimensional array.
        if array.ndim == 0:
            array = np.ma.array(array)
        chunk = Chunk(self.current_keys, array)
        return chunk

    def process_data(self, data):
        self.running_total += np.sum(data, axis=self.axis)


class _MeanStreamsHandler(_AggregationStreamsHandler):
    def __init__(self, array, axis, mdtol):
        # The mdtol argument is not applicable to non-masked arrays
        # so it is ignored.
        super(_MeanStreamsHandler, self).__init__(array, axis)

    def bootstrap(self, processed_chunk_shape):
        self.running_total = np.zeros(processed_chunk_shape,
                                      dtype=self.array.dtype)

    def finalise(self):
        array = self.running_total / self.array.shape[self.axis]
        # Promote array-scalar to 0-dimensional array.
        if array.ndim == 0:
            array = np.array(array)
        chunk = Chunk(self.current_keys, array)
        return chunk

    def process_data(self, data):
        self.running_total += np.sum(data, axis=self.axis)


class _MeanMaskedStreamsHandler(_AggregationStreamsHandler):
    def __init__(self, array, axis, mdtol):
        self._mdtol = mdtol
        super(_MeanMaskedStreamsHandler, self).__init__(array, axis)

    def bootstrap(self, processed_chunk_shape):
        shape = processed_chunk_shape
        self.running_count = np.zeros(shape, dtype=self.array.dtype)
        self.running_masked_count = np.zeros(shape, dtype=self.array.dtype)
        self.running_total = np.zeros(shape, dtype=self.array.dtype)

    def finalise(self):
        # Avoid any runtime-warning for divide by zero.
        mask = self.running_count == 0
        denominator = np.ma.array(self.running_count, mask=mask, dtype=float)
        array = np.ma.array(self.running_total, mask=mask) / denominator
        # Promote array-scalar to 0-dimensional array.
        if array.ndim == 0:
            array = np.ma.array(array)
        # Apply masked/missing data threshold (mdtol).
        if self._mdtol < 1:
            mask_update = np.true_divide(self.running_masked_count,
                                         self.running_masked_count +
                                         self.running_count) > self._mdtol
            array.mask |= mask_update

        chunk = Chunk(self.current_keys, array)
        return chunk

    def process_data(self, data):
        self.running_count += np.ma.count(data, axis=self.axis)
        self.running_masked_count += np.ma.count_masked(data, axis=self.axis)
        self.running_total += np.sum(data, axis=self.axis)


class _StdStreamsHandler(_AggregationStreamsHandler):
    # The algorithm used here preserves numerical accuracy whilst only
    # requiring a single pass, and is taken from:
    # Welford, BP (August 1962). "Note on a Method for Calculating
    # Corrected Sums of Squares and Products".
    # Technometrics 4 (3): 419-420.
    # http://zach.in.tu-clausthal.de/teaching/info_literatur/Welford.pdf

    def __init__(self, array, axis, ddof):
        self.ddof = ddof
        super(_StdStreamsHandler, self).__init__(array, axis)

    def bootstrap(self, processed_chunk_shape):
        self.k = 1
        dtype = (np.zeros(1, dtype=self.array.dtype) / 1.).dtype
        self.q = np.zeros(processed_chunk_shape, dtype=dtype)

    def finalise(self):
        self.q /= (self.k - self.ddof)
        array = np.sqrt(self.q)
        # Promote array-scalar to 0-dimensional array.
        if array.ndim == 0:
            array = np.array(array)
        chunk = Chunk(self.current_keys, array)
        return chunk

    def process_data(self, data):
        data = np.rollaxis(data, self.axis)

        if self.k == 1:
            self.a = data[0].copy()
            data = data[1:]

        for data_slice in data:
            self.k += 1

            # Compute a(k).
            temp = data_slice - self.a
            temp /= self.k
            self.a += temp

            # Compute q(k).
            temp *= temp
            temp *= self.k * (self.k - 1)
            self.q += temp


class _StdMaskedStreamsHandler(_AggregationStreamsHandler):
    # The algorithm used here preserves numerical accuracy whilst only
    # requiring a single pass, and is taken from:
    # Welford, BP (August 1962). "Note on a Method for Calculating
    # Corrected Sums of Squares and Products".
    # Technometrics 4 (3): 419-420.
    # http://zach.in.tu-clausthal.de/teaching/info_literatur/Welford.pdf

    def __init__(self, array, axis, ddof):
        self.ddof = ddof
        super(_StdMaskedStreamsHandler, self).__init__(array, axis)
        self.target_shape = list(self.array.shape)
        del self.target_shape[self.axis]

    def bootstrap(self, processed_chunk_shape):
        dtype = (np.zeros(1, dtype=self.array.dtype) / 1.).dtype
        self.a = np.zeros(processed_chunk_shape, dtype=dtype).flatten()
        self.q = np.zeros(processed_chunk_shape, dtype=dtype).flatten()
        self.running_count = np.zeros(processed_chunk_shape,
                                      dtype=dtype).flatten()
        self.current_shape = processed_chunk_shape

    def finalise(self):
        mask = self.running_count == 0
        denominator = ma.array(self.running_count, mask=mask) - self.ddof
        q = ma.array(self.q, mask=mask) / denominator
        array = ma.sqrt(q)
        array.shape = self.current_shape
        # Promote array-scalar to 0-dimensional array.
        if array.ndim == 0:
            array = np.ma.array(array)
        chunk = Chunk(self.current_keys, array)
        return chunk

    def process_data(self, data):
        data = np.rollaxis(data, self.axis)
        for chunk_slice in data:
            chunk_slice = chunk_slice.flatten()
            bootstrapped = self.running_count != 0
            have_data = ~ma.getmaskarray(chunk_slice)
            chunk_data = ma.array(chunk_slice).filled(0)

            # Bootstrap a(k) where necessary.
            self.a[~bootstrapped] = chunk_data[~bootstrapped]

            self.running_count += have_data

            # Compute a(k).
            do_stuff = bootstrapped & have_data
            temp = ((chunk_data[do_stuff] - self.a[do_stuff]) /
                    self.running_count[do_stuff])
            self.a[do_stuff] += temp

            # Compute q(k).
            temp *= temp
            temp *= (self.running_count[do_stuff] *
                     (self.running_count[do_stuff] - 1))
            self.q[do_stuff] += temp


class _VarStreamsHandler(_StdStreamsHandler):
    def finalise(self):
        chunk = super(_VarStreamsHandler, self).finalise()
        chunk = Chunk(chunk.keys, chunk.data * chunk.data)
        return chunk


class _VarMaskedStreamsHandler(_StdMaskedStreamsHandler):
    def finalise(self):
        chunk = super(_VarMaskedStreamsHandler, self).finalise()
        chunk = Chunk(chunk.keys, chunk.data * chunk.data)
        return chunk


class ComputedArray(Array):
    @abstractproperty
    def sources(self):
        """The tuple of Array instances from which the result is computed."""

    @abstractmethod
    def streams_handler(self, masked):
        """Return a StreamsHandler which can compute the result."""


class _Aggregation(ComputedArray):
    def __init__(self, array, axis,
                 streams_handler_class, masked_streams_handler_class,
                 dtype, kwargs):
        self._array = ensure_array(array)
        self._axis = axis
        self._streams_handler_class = streams_handler_class
        self._masked_streams_handler_class = masked_streams_handler_class
        self._dtype = dtype
        self._kwargs = kwargs

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        shape = list(self._array.shape)
        del shape[self._axis]
        return tuple(shape)

    @property
    def sources(self):
        return (self._array,)

    def _getitem_full_keys(self, keys):
        # Insert an ':' into these keys to get keys for self._array.
        keys = list(keys)
        keys[self._axis:self._axis] = [slice(None)]
        keys = tuple(keys)
        # Reduce the aggregation-axis by the number of prior dimensions that
        # get removed by the indexing operation.
        scalar_axes = map(_is_scalar, keys[:self._axis])
        result_axis = self._axis - __builtin__.sum(scalar_axes)
        return _Aggregation(self._array[keys], result_axis,
                            self._streams_handler_class,
                            self._masked_streams_handler_class,
                            self.dtype,
                            self._kwargs)

    def ndarray(self):
        result, = engine.ndarrays(self)
        return result

    def masked_array(self):
        result, = engine.masked_arrays(self)
        return result

    def streams_handler(self, masked):
        if masked:
            handler_class = self._masked_streams_handler_class
        else:
            handler_class = self._streams_handler_class
        source, = self.sources
        return handler_class(source, self._axis, **self._kwargs)


def _normalise_axis(axis, array):
    # Convert `axis` to None, or a tuple of positive ints, or raise a
    # TypeError/ValueError.
    if axis is None:
        axes = None
    elif _is_scalar(axis):
        axes = (axis,)
    elif (isinstance(axis, collections.Iterable) and
            not isinstance(axis, (basestring, collections.Mapping)) and
            all(map(_is_scalar, axis))):
        axes = tuple(axis)
    else:
        raise TypeError('axis must be None, int, or iterable of ints')
    if axes is not None:
        axes = tuple(axis if axis >= 0 else array.ndim + axis for axis in axes)
        if not all(0 <= axis < array.ndim for axis in axes):
            raise ValueError("'axis' value is out of bounds")
    return axes


def count(a, axis=None):
    """
    Count the non-masked elements of the array along the given axis.

    .. note:: Currently limited to operating on a single axis.

    :param axis: Axis or axes along which the operation is performed.
                 The default (axis=None) is to perform the operation
                 over all the dimensions of the input array.
                 The axis may be negative, in which case it counts from
                 the last to the first axis.
                 If axis is a tuple of ints, the operation is performed
                 over multiple axes.
    :type axis: None, or int, or iterable of ints.
    :return: The Array representing the requested mean.
    :rtype: Array

    """
    axes = _normalise_axis(axis, a)
    if axes is None or len(axes) != 1:
        msg = "This operation is currently limited to a single axis"
        raise AxisSupportError(msg)
    return _Aggregation(a, axes[0],
                        _CountStreamsHandler, _CountMaskedStreamsHandler,
                        np.dtype('i'), {})


def min(a, axis=None):
    """
    Request the minimum of an Array over any number of axes.

    .. note:: Currently limited to operating on a single axis.

    Parameters
    ----------
    a : Array object
        The object whose minimum is to be found.
    axis : None, or int, or iterable of ints
        Axis or axes along which the operation is performed. The default
        (axis=None) is to perform the operation over all the dimensions of the
        input array. The axis may be negative, in which case it counts from
        the last to the first axis. If axis is a tuple of ints, the operation
        is performed over multiple axes.

    Returns
    -------
    out : Array
        The Array representing the requested mean.
    """
    axes = _normalise_axis(axis, a)
    assert axes is not None and len(axes) == 1
    return _Aggregation(a, axes[0],
                        _MinStreamsHandler, _MinMaskedStreamsHandler,
                        a.dtype, {})


def max(a, axis=None):
    """
    Request the maximum of an Array over any number of axes.

    .. note:: Currently limited to operating on a single axis.

    Parameters
    ----------
    a : Array object
        The object whose maximum is to be found.
    axis : None, or int, or iterable of ints
        Axis or axes along which the operation is performed. The default
        (axis=None) is to perform the operation over all the dimensions of the
        input array. The axis may be negative, in which case it counts from
        the last to the first axis. If axis is a tuple of ints, the operation
        is performed over multiple axes.

    Returns
    -------
    out : Array
        The Array representing the requested max.
    """
    axes = _normalise_axis(axis, a)
    assert axes is not None and len(axes) == 1
    return _Aggregation(a, axes[0],
                        _MaxStreamsHandler, _MaxMaskedStreamsHandler,
                        a.dtype, {})


def sum(a, axis=None):
    """
    Request the sum of an Array over any number of axes.

    .. note:: Currently limited to operating on a single axis.

    Parameters
    ----------
    a : Array object
        The object whose summation is to be found.
    axis : None, or int, or iterable of ints
        Axis or axes along which the operation is performed. The default
        (axis=None) is to perform the operation over all the dimensions of the
        input array. The axis may be negative, in which case it counts from
        the last to the first axis. If axis is a tuple of ints, the operation
        is performed over multiple axes.

    Returns
    -------
    out : Array
        The Array representing the requested sum.
    """
    axes = _normalise_axis(axis, a)
    assert axes is not None and len(axes) == 1
    return _Aggregation(a, axes[0],
                        _SumStreamsHandler, _SumMaskedStreamsHandler,
                        a.dtype, {})


def mean(a, axis=None, mdtol=1):
    """
    Request the mean of an Array over any number of axes.

    .. note:: Currently limited to operating on a single axis.

    :param axis: Axis or axes along which the operation is performed.
                 The default (axis=None) is to perform the operation
                 over all the dimensions of the input array.
                 The axis may be negative, in which case it counts from
                 the last to the first axis.
                 If axis is a tuple of ints, the operation is performed
                 over multiple axes.
    :type axis: None, or int, or iterable of ints.
    :param float mdtol: Tolerance of missing data. The value in each
                        element of the resulting array will be masked if the
                        fraction of masked data contributing to that element
                        exceeds mdtol. mdtol=0 means no missing data is
                        tolerated while mdtol=1 will mean the resulting
                        element will be masked if and only if all the
                        contributing elements of the source array are masked.
                        Defaults to 1.
    :return: The Array representing the requested mean.
    :rtype: Array

    """
    axes = _normalise_axis(axis, a)
    if axes is None or len(axes) != 1:
        msg = "This operation is currently limited to a single axis"
        raise AxisSupportError(msg)
    dtype = (np.array([0], dtype=a.dtype) / 1.).dtype
    kwargs = dict(mdtol=mdtol)
    return _Aggregation(a, axes[0],
                        _MeanStreamsHandler, _MeanMaskedStreamsHandler,
                        dtype, kwargs)


def std(a, axis=None, ddof=0):
    """
    Request the standard deviation of an Array over any number of axes.

    .. note:: Currently limited to operating on a single axis.

    :param axis: Axis or axes along which the operation is performed.
                 The default (axis=None) is to perform the operation
                 over all the dimensions of the input array.
                 The axis may be negative, in which case it counts from
                 the last to the first axis.
                 If axis is a tuple of ints, the operation is performed
                 over multiple axes.
    :type axis: None, or int, or iterable of ints.
    :param int ddof: Delta Degrees of Freedom. The divisor used in
                     calculations is N - ddof, where N represents the
                     number of elements. By default ddof is zero.
    :return: The Array representing the requested standard deviation.
    :rtype: Array

    """
    axes = _normalise_axis(axis, a)
    if axes is None or len(axes) != 1:
        msg = "This operation is currently limited to a single axis"
        raise AxisSupportError(msg)
    dtype = (np.array([0], dtype=a.dtype) / 1.).dtype
    return _Aggregation(a, axes[0],
                        _StdStreamsHandler, _StdMaskedStreamsHandler,
                        dtype, dict(ddof=ddof))


def var(a, axis=None, ddof=0):
    """
    Request the variance of an Array over any number of axes.

    .. note:: Currently limited to operating on a single axis.

    :param axis: Axis or axes along which the operation is performed.
                 The default (axis=None) is to perform the operation
                 over all the dimensions of the input array.
                 The axis may be negative, in which case it counts from
                 the last to the first axis.
                 If axis is a tuple of ints, the operation is performed
                 over multiple axes.
    :type axis: None, or int, or iterable of ints.
    :param int ddof: Delta Degrees of Freedom. The divisor used in
                     calculations is N - ddof, where N represents the
                     number of elements. By default ddof is zero.
    :return: The Array representing the requested variance.
    :rtype: Array

    """
    axes = _normalise_axis(axis, a)
    if axes is None or len(axes) != 1:
        msg = "This operation is currently limited to a single axis"
        raise AxisSupportError(msg)
    dtype = (np.array([0], dtype=a.dtype) / 1.).dtype
    return _Aggregation(a, axes[0],
                        _VarStreamsHandler, _VarMaskedStreamsHandler,
                        dtype, dict(ddof=ddof))


class _ElementwiseStreamsHandler(_StreamsHandler):
    def __init__(self, sources, operator):
        self.sources = sources
        self.operator = operator

    def finalise(self):
        pass

    def input_iteration_order(self, iteration_order):
        return iteration_order

    def process_chunks(self, chunks):
        array = self.operator(*[chunk.data for chunk in chunks])
        chunk = Chunk(chunks[0].keys, array)
        return chunk


class _Elementwise(ComputedArray):
    """"
    Represents an elementwise operation applied to one or two input arrays.

    Elementwise operations are defined by passing an appropriate function,
    and optionally an equivalent masked_array function. Examples of elementwise
    operations included addition (two inputs) and cos (one input).

    """
    def __init__(self, array1, array2, numpy_op, ma_op=None):
        array1 = ensure_array(array1)

        expected_n_arrays = None
        if isinstance(numpy_op, np.ufunc):
            expected_n_arrays = numpy_op.nin
            if numpy_op.nout != 1:
                raise TypeError('Dual output Elementwise ufuncs not yet '
                                'supported.')

        if array2 is None:
            # Unary elementwise.

            if expected_n_arrays is not None and expected_n_arrays != 1:
                # Trigger an exception (TypeError).
                numpy_op(np.array([1], dtype=array1.dtype), None)

            # Catch warnings for dtype calculation. This typically occurs with
            # operations which don't work well with 1 (e.g. arctanh).
            with warnings.catch_warnings(record=True):
                dtype = numpy_op(np.array([1], dtype=array1.dtype)).dtype
        else:
            if expected_n_arrays is not None and expected_n_arrays != 2:
                # Trigger an exception (TypeError).
                numpy_op(np.array([1], dtype=array1.dtype))

            # Keep initial_array2 for later type checking.
            initial_array2 = array2

            # Dual input elementwise
            array2 = ensure_array(array2)

            # Broadcast both arrays to the full broadcast shape.
            # TypeError will be raised if not broadcastable.
            array1, array2 = BroadcastArray.broadcast_arrays(array1, array2)

            # Explicitly check the type, using is, not isinstance;
            # this is to trap the case where a Python int of float is provided,
            # rather than a numpy int or float.  The type promotion is handled
            # differently, for consistency with numpy (1.8 & 1.9) behaviour.
            if type(initial_array2) is int or type(initial_array2) is float:
                second_array = initial_array2
            else:
                second_array = np.ones(1, dtype=array2.dtype)

            # Type-promotion - The resultant dtype depends on both the array
            # dtypes and the operation. Avoid using np.find_common_dtype(),
            # here, as integer division yields a float, whereas standard type
            # coercion rules with find_common_dtype yields an integer.
            dtype = numpy_op(np.ones(1, dtype=array1.dtype),
                             second_array).dtype

        self._dtype = dtype
        self._array1 = array1
        self._array2 = array2
        self._numpy_op = numpy_op
        self._ma_op = ma_op

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._array1.shape

    @property
    def sources(self):
        if self._array2 is None:
            result = (self._array1, )
        else:
            result = (self._array1, self._array2)
        return result

    def _getitem_full_keys(self, keys):
        if self._array2 is None:
            result = _Elementwise(self._array1[keys], None,
                                  self._numpy_op, self._ma_op)
        else:
            result = _Elementwise(self._array1[keys], self._array2[keys],
                                  self._numpy_op, self._ma_op)
        return result

    def ndarray(self):
        np_operands = ndarrays(self.sources)
        result = self._numpy_op(*np_operands)
        return result

    def masked_array(self):
        if self._ma_op is None:
            raise TypeError('No {} operation defined for masked arrays.'
                            ''.format(self._numpy_op.__name__))
        ma_operands = masked_arrays(self.sources)
        result = self._ma_op(*ma_operands)
        return result

    def streams_handler(self, masked):
        if masked:
            if self._ma_op is None:
                raise TypeError('No {} operation defined for masked arrays.'
                                ''.format(self._numpy_op.__name__))
            operator = self._ma_op
        else:
            operator = self._numpy_op
        return _ElementwiseStreamsHandler(self.sources, operator)


def _unary_fn_wrapper(name, function_to_wrap, masked_equivalent=None,
                      fn_name=None):
    @functools.wraps(function_to_wrap, assigned=('__name__', '__doc__'))
    def wrapped_function(a):
        return _Elementwise(a, None, function_to_wrap, masked_equivalent)
    doc_str = ("Return the elementwise evaluation of {}(a) "
               "as another Array.".format(name))
    wrapped_function.__name__ = fn_name or function_to_wrap.__name__
    wrapped_function.__doc__ = doc_str
    return wrapped_function


def _dual_input_fn_wrapper(name, function_to_wrap, masked_equivalent=None,
                           fn_name=None):
    @functools.wraps(function_to_wrap, assigned=('__name__', '__doc__'))
    def wrapped_function(a, b):
        return _Elementwise(a, b, function_to_wrap,
                            masked_equivalent)
    doc_str = ("Return the elementwise evaluation of {}(a, b) "
               "as another Array.".format(name))
    wrapped_function.__name__ = fn_name or function_to_wrap.__name__
    wrapped_function.__doc__ = doc_str
    return wrapped_function


def _ufunc_wrapper(ufunc):
    """
    A function to generate the top level biggus ufunc wrappers.

    """
    if not isinstance(ufunc, np.ufunc):
        raise TypeError('{} is not a ufunc'.format(ufunc))

    name = ufunc.__name__
    # Get hold of the masked array equivalent, if it exists.
    ma_ufunc = getattr(np.ma, name, None)
    if ufunc.nin == 2 and ufunc.nout == 1:
        func = _dual_input_fn_wrapper('np.{}'.format(name), ufunc, ma_ufunc,
                                      name)
    elif ufunc.nin == 1 and ufunc.nout == 1:
        func = _unary_fn_wrapper('np.{}'.format(name), ufunc, ma_ufunc,
                                 name)
    else:
        raise ValueError('Unsupported ufunc {!r} with {} input arrays & {} '
                         'output arrays.'.format(name, ufunc.nin, ufunc.nout))
    return func


# Single argument math operations.
negative = _ufunc_wrapper(np.negative)
absolute = _ufunc_wrapper(np.absolute)
rint = _ufunc_wrapper(np.rint)
sign = _ufunc_wrapper(np.sign)
conj = _ufunc_wrapper(np.conj)
exp = _ufunc_wrapper(np.exp)
exp2 = _ufunc_wrapper(np.exp2)
log = _ufunc_wrapper(np.log)
log2 = _ufunc_wrapper(np.log2)
log10 = _ufunc_wrapper(np.log10)
expm1 = _ufunc_wrapper(np.expm1)
sqrt = _ufunc_wrapper(np.sqrt)
square = _ufunc_wrapper(np.square)
reciprocal = _ufunc_wrapper(np.reciprocal)


# Dual argument math operations.
add = _ufunc_wrapper(np.add)
sub = _ufunc_wrapper(np.subtract)
subtract = _ufunc_wrapper(np.subtract)
multiply = _ufunc_wrapper(np.multiply)
floor_divide = _ufunc_wrapper(np.floor_divide)
true_divide = _ufunc_wrapper(np.true_divide)
divide = _ufunc_wrapper(np.divide)
power = _ufunc_wrapper(np.power)


# Single argument trigonometric functions.
sin = _ufunc_wrapper(np.sin)
cos = _ufunc_wrapper(np.cos)
tan = _ufunc_wrapper(np.tan)
arcsin = _ufunc_wrapper(np.arcsin)
arccos = _ufunc_wrapper(np.arccos)
arctan = _ufunc_wrapper(np.arctan)
sinh = _ufunc_wrapper(np.sinh)
cosh = _ufunc_wrapper(np.cosh)
tanh = _ufunc_wrapper(np.tanh)
arcsinh = _ufunc_wrapper(np.arcsinh)
arccosh = _ufunc_wrapper(np.arccosh)
arctanh = _ufunc_wrapper(np.arctanh)
deg2rad = _ufunc_wrapper(np.deg2rad)
rad2deg = _ufunc_wrapper(np.rad2deg)


# Dual argument trigonometric functions.
arctan2 = _ufunc_wrapper(np.arctan2)
hypot = _ufunc_wrapper(np.hypot)


# Bit-twiddling functions.
bitwise_and = _ufunc_wrapper(np.bitwise_and)
bitwise_or = _ufunc_wrapper(np.bitwise_or)
bitwise_xor = _ufunc_wrapper(np.bitwise_xor)
invert = _ufunc_wrapper(np.invert)
left_shift = _ufunc_wrapper(np.left_shift)
right_shift = _ufunc_wrapper(np.right_shift)


# Comparison functions.
greater = _ufunc_wrapper(np.greater)
greater_equal = _ufunc_wrapper(np.greater_equal)
less = _ufunc_wrapper(np.less)
less_equal = _ufunc_wrapper(np.less_equal)
not_equal = _ufunc_wrapper(np.not_equal)
equal = _ufunc_wrapper(np.equal)
logical_and = _ufunc_wrapper(np.logical_and)
logical_or = _ufunc_wrapper(np.logical_or)
logical_xor = _ufunc_wrapper(np.logical_xor)
logical_not = _ufunc_wrapper(np.logical_not)
maximum = _ufunc_wrapper(np.maximum)
minimum = _ufunc_wrapper(np.minimum)
fmax = _ufunc_wrapper(np.fmax)
fmin = _ufunc_wrapper(np.fmin)


# Floating functions.
isreal = _unary_fn_wrapper('np.isreal', np.isreal)
iscomplex = _unary_fn_wrapper('np.iscomplex', np.iscomplex)
isinf = _unary_fn_wrapper('np.isinf', np.isinf)
isnan = _unary_fn_wrapper('np.isnan', np.isnan)
signbit = _unary_fn_wrapper('np.signbit', np.signbit)
copysign = _dual_input_fn_wrapper('np.copysign', np.copysign)
nextafter = _dual_input_fn_wrapper('np.nextafter', np.nextafter)
# modf = _unary_fn_wrapper('np.modf', np.modf)  # Needs 2 arrays out.
ldexp = _dual_input_fn_wrapper('np.ldexp', np.ldexp)
# frexp = _unary_fn_wrapper('np.frexp', np.frexp)  # Needs 2 arrays out.
fmod = _dual_input_fn_wrapper('np.fmod', np.fmod)
floor = _ufunc_wrapper(np.floor)
ceil = _ufunc_wrapper(np.ceil)
trunc = _ufunc_wrapper(np.trunc)


def _sliced_shape(shape, keys):
    """
    Returns the shape that results from slicing an array of the given
    shape by the given keys.

    >>> _sliced_shape(shape=(52350, 70, 90, 180),
    ...               keys=(np.newaxis, slice(None, 10), 3,
    ...                     slice(None), slice(2, 3)))
    (1, 10, 90, 1)

    """
    keys = _full_keys(keys, len(shape))

    sliced_shape = []
    shape_dim = -1
    for key in keys:
        shape_dim += 1
        if _is_scalar(key):
            continue
        elif isinstance(key, slice):
            size = len(range(*key.indices(shape[shape_dim])))
            sliced_shape.append(size)
        elif isinstance(key, np.ndarray) and key.dtype == np.dtype('bool'):
            # Numpy boolean indexing.
            sliced_shape.append(__builtin__.sum(key))
        elif isinstance(key, (tuple, np.ndarray)):
            sliced_shape.append(len(key))
        elif key is np.newaxis:
            shape_dim -= 1
            sliced_shape.append(1)
        else:
            raise ValueError('Invalid indexing object "{}"'.format(key))

    sliced_shape = tuple(sliced_shape)
    return sliced_shape


def _full_keys(keys, ndim):
    """
    Given keys such as those passed to ``__getitem__`` for an
    array of ndim, return a fully expanded tuple of keys.

    In all instances, the result of this operation should follow:

        array[keys] == array[_full_keys(keys, array.ndim)]

    """
    if not isinstance(keys, tuple):
        keys = (keys,)

    # Make keys mutable, and take a copy.
    keys = list(keys)

    # Count the number of keys which actually slice a dimension.
    n_keys_non_newaxis = len([key for key in keys if key is not np.newaxis])

    # Numpy allows an extra dimension to be an Ellipsis, we remove it here
    # if Ellipsis is in keys, if this doesn't trigger we will raise an
    # IndexError.
    is_ellipsis = [key is Ellipsis for key in keys]
    if n_keys_non_newaxis - 1 >= ndim and any(is_ellipsis):
        # Remove the left-most Ellipsis, as numpy does.
        keys.pop(is_ellipsis.index(True))
        n_keys_non_newaxis -= 1

    if n_keys_non_newaxis > ndim:
        raise IndexError('Dimensions are over specified for indexing.')

    lh_keys = []
    # Keys, with the last key first.
    rh_keys = []

    take_from_left = True
    while keys:
        if take_from_left:
            next_key = keys.pop(0)
            keys_list = lh_keys
        else:
            next_key = keys.pop(-1)
            keys_list = rh_keys

        if next_key is Ellipsis:
            next_key = slice(None)
            take_from_left = not take_from_left
        keys_list.append(next_key)

    middle = [slice(None)] * (ndim - n_keys_non_newaxis)
    return tuple(lh_keys + middle + rh_keys[::-1])


def ensure_array(array):
    """
    Assert that the given array is an Array subclass (or numpy array).

    If the given array is a numpy.ndarray an appropriate NumpyArrayAdapter
    instance is created, otherwise the passed array must be a subclass of
    :class:`Array` else a TypeError will be raised.

    """
    if not isinstance(array, Array):
        if isinstance(array, np.ndarray):
            array = NumpyArrayAdapter(array)
        elif np.isscalar(array):
            array = ConstantArray([], array)
        else:
            raise TypeError('The given array should be a `biggus.Array` '
                            'instance, got {}.'.format(type(array)))
    return array


def size(array):
    """
    Return a human-readable description of the number of bytes required
    to store the data of the given array.

    For example::

        >>> array.nbytes
        14000000
        >> biggus.size(array)
        '13.35 MiB'

    Parameters
    ----------
    array : array-like object
        The array object must provide an `nbytes` property.

    Returns
    -------
    out : str
        The Array representing the requested mean.

    """
    nbytes = array.nbytes
    if nbytes < (1 << 10):
        size = '{} B'.format(nbytes)
    elif nbytes < (1 << 20):
        size = '{:.02f} KiB'.format(nbytes / (1 << 10))
    elif nbytes < (1 << 30):
        size = '{:.02f} MiB'.format(nbytes / (1 << 20))
    elif nbytes < (1 << 40):
        size = '{:.02f} GiB'.format(nbytes / (1 << 30))
    else:
        size = '{:.02f} TiB'.format(nbytes / (1 << 40))
    return size
