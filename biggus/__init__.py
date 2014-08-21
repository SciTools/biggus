# (C) British Crown Copyright 2012 - 2014, Met Office
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
import itertools
import threading
import Queue

import numpy as np
import numpy.ma as ma


__version__ = '0.7.0'


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
            all_cuts = _all_slices_inner(self.array.dtype.itemsize,
                                         self.array.shape,
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
        An Array entirely filled with ones.

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
        keys = self._normalise_keys(keys)
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
        shape = _sliced_shape(self.concrete.shape, self._keys)
        return shape

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

    def __getitem__(self, keys):
        keys = self._normalise_keys(keys)

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


class ArrayStack(Array):
    """
    An Array made from a homogeneous array of other Arrays.

    """
    def __init__(self, stack):
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
            ok = (array.shape == item_shape and array.dtype == dtype and
                  fill_value_ok(array))
            if not ok:
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
        keys = self._normalise_keys(keys)

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
            if not fill_value_ok(tile):
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
        keys = self._normalise_keys(keys)

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
    return engine.ndarrays(*arrays)


#: The maximum number of bytes to allow when processing an array in
#: "bite-size" chunks. The value has been empirically determined to
#: provide vaguely near optimal performance under certain conditions.
MAX_CHUNK_SIZE = 8 * 1024 * 1024


def _all_slices(array):
    return _all_slices_inner(array.dtype.itemsize, array.shape)


def _all_slices_inner(item_size, shape, always_slices=False):
    # Return the slices for each dimension which ensure complete
    # coverage by chunks no larger than MAX_CHUNK_SIZE.
    # e.g. For a float32 array of shape (100, 768, 1024) the slices are:
    #   (0, 1, 2, ..., 99),
    #   (slice(0, 256), slice(256, 512), slice(512, 768)),
    #   (slice(None)
    nbytes = item_size
    all_slices = []
    for i, size in reversed(list(enumerate(shape))):
        if size * nbytes <= MAX_CHUNK_SIZE:
            slices = (slice(None),)
        elif nbytes > MAX_CHUNK_SIZE:
            if always_slices:
                slices = [slice(i, i + 1) for i in range(size)]
            else:
                slices = range(size)
        else:
            step = MAX_CHUNK_SIZE // nbytes
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


class _StreamsHandler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def finalise(self):
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
    def bootstrap(self, data):
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
        if keys != self.current_keys:
            shape = list(chunk.data.shape)
            del shape[self.axis]
            self.current_shape = shape
            if self.current_keys is not None:
                result = self.finalise()
            self.bootstrap(shape)
            self.current_keys = keys
        self.process_data(chunk.data)
        return result

    @abstractmethod
    def process_data(self, data):
        pass


class _CountStreamsHandler(_AggregationStreamsHandler):
    def bootstrap(self, shape):
        self.current_shape = shape
        self.running_count = 0

    def finalise(self):
        count = np.ones(self.current_shape, dtype='i') * self.running_count
        chunk = Chunk(self.current_keys, count)
        return chunk

    def process_data(self, data):
        self.running_count += data.shape[self.axis]


class _CountMaskedStreamsHandler(_AggregationStreamsHandler):
    def bootstrap(self, shape):
        self.running_count = np.zeros(shape, dtype='i')

    def finalise(self):
        chunk = Chunk(self.current_keys, self.running_count)
        return chunk

    def process_data(self, data):
        self.running_count += np.ma.count(data, axis=self.axis)


class _MinStreamsHandler(_AggregationStreamsHandler):
    def bootstrap(self, shape):
        self.result = np.zeros(shape, dtype=self.array.dtype)

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
    def bootstrap(self, shape):
        self.result = np.zeros(shape, dtype=self.array.dtype)

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
    def bootstrap(self, shape):
        self.result = np.zeros(shape, dtype=self.array.dtype)

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
    def bootstrap(self, shape):
        self.result = np.zeros(shape, dtype=self.array.dtype)

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
    def bootstrap(self, shape):
        self.running_total = np.zeros(shape, dtype=self.array.dtype)

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
    def bootstrap(self, shape):
        self.running_total = np.ma.zeros(shape, dtype=self.array.dtype)

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

    def bootstrap(self, shape):
        self.running_total = np.zeros(shape, dtype=self.array.dtype)

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

    def bootstrap(self, shape):
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

    def bootstrap(self, shape):
        self.k = 1
        dtype = (np.zeros(1, dtype=self.array.dtype) / 1.).dtype
        self.q = np.zeros(shape, dtype=dtype)

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

    def bootstrap(self, shape):
        dtype = (np.zeros(1, dtype=self.array.dtype) / 1.).dtype
        self.a = np.zeros(shape, dtype=dtype).flatten()
        self.q = np.zeros(shape, dtype=dtype).flatten()
        self.running_count = np.zeros(shape, dtype=dtype).flatten()

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

    def __getitem__(self, keys):
        keys = self._normalise_keys(keys)
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
    def __init__(self, array1, array2, numpy_op, ma_op):
        array1 = ensure_array(array1)
        array2 = ensure_array(array2)

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

    @property
    def sources(self):
        return (self._array1, self._array2)

    def __getitem__(self, keys):
        keys = self._normalise_keys(keys)
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

    def streams_handler(self, masked):
        if masked:
            operator = self._ma_op
        else:
            operator = self._numpy_op
        return _ElementwiseStreamsHandler(self.sources, operator)


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
    for size, key in itertools.izip_longest(shape, keys):
        if _is_scalar(key):
            continue
        elif isinstance(key, slice):
            size = len(range(*key.indices(size)))
            sliced_shape.append(size)
        elif isinstance(key, np.ndarray) and key.dtype == np.dtype('bool'):
            # Numpy boolean indexing.
            sliced_shape.append(__builtin__.sum(key))
        elif isinstance(key, (tuple, np.ndarray)):
            sliced_shape.append(len(key))
        else:
            sliced_shape.append(size)
    sliced_shape = tuple(sliced_shape)
    return sliced_shape


def ensure_array(array):
    """
    Assert that the given array is an Array subclass (or numpy array).

    If the given array is a numpy.ndarray an appropriate NumpyArrayAdapter
    instance is created, otherwise the passed array must be a subclass of
    :class:`Array` else a TypeError will be raised.

    """
    if isinstance(array, np.ndarray):
        array = NumpyArrayAdapter(array)

    elif not isinstance(array, Array):
        raise TypeError('The given array should be a `biggus.Array` '
                        'instance, got {}.'.format(type(array)))
    return array
