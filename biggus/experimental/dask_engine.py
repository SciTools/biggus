# (C) British Crown Copyright 2017, Met Office
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
An engine that produces a dask graph from the biggus expression.

Note: This is *not* an interface on dask.array, it is an entirely different
implementation of the dask.array like functionality using existing biggus
capabilities (which were predecessors of ``dask.array``).

Graph visualisation usage:

    import biggus
    from biggus.experimental.dask_engine import DaskEngine
    import dask.dot

    e = DaskEngine()

    # The biggus expression.
    a = biggus.zeros([200, 1000, 200])
    arr = biggus.mean(a + 1, axis=0)

    # The arrays to graph.
    arrays = [a, arr]
    dask_graph = e.graph(*arrays)

    g = dask.dot.dot_graph(dask_graph, 'my_dask.png')


To compute the graph, simply call the ``Engine.ndarrays`` method.


"""

try:
    from itertools import izip_longest as zip_longest
except ImportError:
    from itertools import zip_longest
import uuid

import biggus
from biggus._init import Engine
import biggus.experimental.key_grouper as key_grouper


def _filterfalse(predicate, iterable):
    # _filterfalse(lambda x: x%2, range(10)) --> 0 2 4 6 8
    if predicate is None:
        predicate = bool
    for x in iterable:
        if not predicate(x):
            yield x


def _unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # _unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # _unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in _filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def _array_id(array, iteration_order=None, masked=False):
    if iteration_order is None:
        iteration_order = range(array.ndim)
    result = '{}array {}\n\n(id: {})'.format('[masked]' if masked else '',
                                             array.shape, id(array))
    return result


def _groups_of_size(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks."""
    # _groups_of_size('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def slice_repr(slice_instance):
    """
    Turn things like `slice(None, 2, -1)` into `:2:-1`.

    """
    if not isinstance(slice_instance, slice):
        raise TypeError('Unhandled type {}'.format(type(slice_instance)))
    start = slice_instance.start or ''
    stop = slice_instance.stop or ''
    step = slice_instance.step or ''

    msg = '{}:'.format(start)
    if stop:
        msg += '{}'.format(stop)
        if step:
            msg += ':'
    if step:
        msg += '{}'.format(step)
    return msg


class DaskGroup(biggus._init.AllThreadedEngine.Group):
    def __init__(self, arrays):
        self.arrays = arrays
        self._node_cache = {}

    @staticmethod
    def biggus_chunk(chunk_key, biggus_array, masked):
        """
        A function that lazily evaluates a biggus.Chunk. This is useful for
        passing through as a dask task so that we don't have to compute the
        chunk in order to compute the graph.

        """
        if masked:
            array = biggus_array.masked_array()
        else:
            array = biggus_array.ndarray()

        return biggus._init.Chunk(chunk_key, array)

    @staticmethod
    def create_chunks_handler_fn(handler, n_sources, nicename):
        def produce_chunks(produced_keys, *all_chunks):
            all_chunks = list(_groups_of_size(all_chunks, n_sources))
            process_result = None
            for chunks in all_chunks:
                process_result = handler.process_chunks(chunks)
            result = handler.finalise()
            if result is None:
                if process_result is None:
                    raise RuntimeError('No result to process.')
                # Itself returns a chunk.
                return process_result
            else:
                return result
        produce_chunks.__name__ = nicename
        return produce_chunks

    def _make_stream_handler_nodes(self, dsk_graph, array, iteration_order,
                                   masked):
        """
        Produce task graph entries for an array that comes from a biggus
        StreamsHandler.

        This is essentially every type of array that isn't already a thing on
        disk/in-memory. StreamsHandler arrays include all aggregations and
        elementwise operations.

        """
        nodes = {}
        handler = array.streams_handler(masked)
        input_iteration_order = handler.input_iteration_order(iteration_order)

        def input_keys_transform(input_array, keys):
            if hasattr(input_array, 'streams_handler'):
                handler = input_array.streams_handler(masked)
                # Get the transformer of the input array, and apply it to the
                # keys.
                input_transformer = getattr(handler,
                                            'output_keys', None)
                if input_transformer is not None:
                    keys = input_transformer(keys)
            return keys

        sources_keys = []
        sources_chunks = []
        for input_array in array.sources:
            # Bring together all chunks that influence the same part of this
            # (resultant) array.
            source_chunks_by_key = {}
            sources_chunks.append(source_chunks_by_key)
            source_keys = []
            sources_keys.append(source_keys)

            # Make nodes for the source arrays (if they don't already exist)
            # before we do anything else.
            input_nodes = self._make_nodes(dsk_graph, input_array,
                                           input_iteration_order, masked)

            for chunk_id, task in input_nodes.items():
                chunk_keys = task[1]
                t_keys = chunk_keys
                t_keys = input_keys_transform(array, t_keys)
                source_keys.append(t_keys)
                this_key = str(t_keys)
                source_chunks_by_key.setdefault(this_key,
                                                []).append([chunk_id, task])

        sources_keys_grouped = key_grouper.group_keys(array.shape,
                                                      *sources_keys)
        for slice_group, sources_keys_group in sources_keys_grouped.items():
            # Each group is entirely independent and can have its own task
            # without knowledge of results from items in other groups.

            t_keys = tuple(slice(*slice_tuple) for slice_tuple in slice_group)

            all_chunks = []
            for source_keys, source_chunks_by_key in zip(sources_keys_group,
                                                         sources_chunks):
                dependencies = tuple(
                        the_id
                        for keys in source_keys
                        for the_id, task in source_chunks_by_key[str(keys)])
                # Uniquify source_keys, but keep the order.
                dependencies = tuple(_unique_everseen(dependencies))

                def normalize_keys(keys, shape):
                    result = []
                    for key, dim_length in zip(keys, shape):
                        result.append(key_grouper.normalize_slice(key,
                                                                  dim_length))
                    return tuple(result)

                # If we don't have the same chunks for all inputs then we
                # should combine them before passing them on to the handler.
                # TODO: Fix slice equality to deal with 0 and None etc.
                if not all(t_keys == normalize_keys(keys, array.shape)
                           for keys in source_keys):
                    combined = self.collect(array[t_keys], masked, chunk=True)
                    new_task = (combined, ) + dependencies
                    new_id = ('chunk shape: {}\n\n{}'
                              ''.format(array[t_keys].shape, uuid.uuid()))
                    dsk_graph[new_id] = new_task
                    dependencies = (new_id, )

                all_chunks.append(dependencies)

            pivoted = all_chunks

            sub_array = array[t_keys]
            handler = sub_array.streams_handler(masked)
            name = getattr(handler, 'nice_name', handler.__class__.__name__)

            if hasattr(handler, 'axis'):
                name += '\n(axis={})'.format(handler.axis)
            # For ElementwiseStreams handlers, use the function that they wrap
            # (e.g "add")
            if hasattr(handler, 'operator'):
                name = handler.operator.__name__

            n_sources = len(array.sources)
            handler_of_chunks_fn = self.create_chunks_handler_fn(handler,
                                                                 n_sources,
                                                                 name)

            shape = sub_array.shape
            if all(key == slice(None) for key in t_keys):
                subset = ''
            else:
                pretty_index = ', '.join(map(slice_repr, t_keys))
                subset = 'target subset [{}]\n'.format(pretty_index)

            # Flatten out the pivot so that dask can dereferences the IDs
            source_chunks = [item for sublist in pivoted for item in sublist]
            task = tuple([handler_of_chunks_fn, t_keys] + source_chunks)
            shape_repr = ', '.join(map(str, shape))
            chunk_id = 'chunk shape: ({})\n\n{}{}'.format(shape_repr,
                                                          subset,
                                                          uuid.uuid4())
            assert chunk_id not in dsk_graph
            dsk_graph[chunk_id] = task
            nodes[chunk_id] = task
        return nodes

    @staticmethod
    def lazy_chunk_creator(name):
        """
        Create a lazy chunk creating function with a nice name that is suitable
        for representation in a dask graph.

        """
        # TODO: Could this become a LazyChunk class?
        def biggus_chunk(chunk_key, biggus_array, masked):
            """
            A function that lazily evaluates a biggus.Chunk. This is useful for
            passing through as a dask task so that we don't have to compute the
            chunk in order to compute the graph.

            """
            if masked:
                array = biggus_array.masked_array()
            else:
                array = biggus_array.ndarray()

            return biggus._init.Chunk(chunk_key, array)
        biggus_chunk.__name__ = name
        return biggus_chunk

    def _make_nodes(self, dsk_graph, array, iteration_order, masked,
                    top=False):
        """
        Recursive function that returns the dask items for the given array.

        NOTE: Currently assuming that all tasks are a tuple, with the second
        item being the keys used to index the source of the respective input
        array.

        """
        cache_key = _array_id(array, iteration_order, masked)
        # By the end of this function Nodes will be a dictionary with one item
        # per chunk to be processed for this array.
        nodes = self._node_cache.get(cache_key, None)

        if nodes is None:
            if hasattr(array, 'streams_handler'):
                nodes = self._make_stream_handler_nodes(dsk_graph, array,
                                                        iteration_order,
                                                        masked)
            else:
                nodes = {}
                chunks = []

                name = '{}\n{}'.format(array.__class__.__name__, array.shape)
                biggus_chunk_func = self.lazy_chunk_creator(name)

                chunk_index_gen = biggus._init.ProducerNode.chunk_index_gen
                for chunk_key in chunk_index_gen(array.shape,
                                                 iteration_order[::-1]):
                    biggus_array = array[chunk_key]
                    pretty_key = ', '.join(map(slice_repr, chunk_key))
                    chunk_id = ('chunk shape: {}\nsource key: [{}]\n\n{}'
                                ''.format(biggus_array.shape, pretty_key,
                                          uuid.uuid4()))
                    task = (biggus_chunk_func, chunk_key, biggus_array, masked)
                    chunks.append(task)
                    assert chunk_id not in dsk_graph
                    dsk_graph[chunk_id] = task
                    nodes[chunk_id] = task
            self._node_cache[cache_key] = nodes
        return nodes

    @staticmethod
    def collect(array, masked, chunk=False, name=None):
        def gather(*all_chunks):
            # We make the NdarrayNode inside the calling function as it is this
            # that ultimately we want. TODO: Turn this into a biggus.Array
            # subclass concept.
            result_node = biggus._init.NdarrayNode(array, masked)
            for chunks in all_chunks:
                # TODO: Factor it so that this isn't necessary...
                if isinstance(chunks, biggus._init.Chunk):
                    chunks = [chunks]
                result_node.process_chunks(chunks)
            result_array = result_node.result
            if chunk:
                return biggus._init.Chunk(tuple(slice(None)
                                                for _ in result_array.shape),
                                          result_array)
            return result_array
        if name:
            gather.__name__ = name
        return gather

    def dask(self, masked=False):
        # Construct nodes starting from the producers.
        dsk_graph = {}
        for array in self.arrays:
            self.dask_task(dsk_graph, array, masked=masked, top=True)
            array_id_val = _array_id(array, masked=masked)
            dependencies = tuple(self._node_cache[array_id_val].keys())
            if array_id_val not in dsk_graph:
                dsk_graph[array_id_val] = (self.collect(array, masked),
                                           ) + dependencies

        return dsk_graph

    def dask_task(self, dsk_graph, array, masked=False, top=False):
        return self._make_nodes(dsk_graph, array, range(array.ndim), masked)


class DaskEngine(Engine):
    """
    An engine that converts the biggus expression graph into a
    dask task graph.

    """
    def __init__(self, dask_getter=None):
        if dask_getter is None:
            import dask.threaded
            dask_getter = dask.threaded.get
        self.dask_getter = dask_getter

    def _daskify(self, arrays, masked=False):
        return DaskGroup(arrays).dask(masked)

    def masked_arrays(self, *arrays):
        ids = [_array_id(array, masked=True) for array in arrays]
        return self.dask_getter(self._daskify(arrays, masked=True), ids)

    def ndarrays(self, *arrays):
        ids = [_array_id(array, masked=False) for array in arrays]
        return self.dask_getter(self._daskify(arrays, masked=False), ids)

    def graph(self, *arrays):
        # TODO: Return a dask.base.Base instance (of this dict). We then get
        # nice methods...
        return self._daskify(arrays)
