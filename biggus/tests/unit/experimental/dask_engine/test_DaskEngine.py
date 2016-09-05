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
"""Unit tests for `biggus.experimental.dask_engine.DaskEngine`."""

from __future__ import absolute_import, division, print_function
from six.moves import (filter, input, map, range, zip)  # noqa

from contextlib import contextmanager
import unittest

import numpy as np

import biggus
from biggus.experimental.dask_engine import DaskEngine, DaskGroup


class Test_key_names(unittest.TestCase):
    def test_key_names(self):
        a = biggus.ConstantArray((2, 5))
        expr = biggus.mean(a + a, axis=1)
        graph = DaskEngine().graph(expr)
        func_names = {task[0].__name__ for task in graph.values()
                      if callable(task[0])}
        expected = {'gather', 'add',
                    'ConstantArray\n(2, 5)',
                    'mean\n(axis=1)'}
        self.assertEqual(expected, func_names)

    def test_repeat_arrays(self):
        a = biggus.ConstantArray((2, 5))
        expr = (a/3.) + a - (a * 2)
        graph = DaskEngine().graph(expr)
        func_names = {task[0].__name__ for task in graph.values()
                      if callable(task[0])}
        expected = {'gather', 'add', 'subtract', 'true_divide',
                    'multiply', 'ConstantArray\n(2, 5)',
                    'BroadcastArray\n(2, 5)'}
        self.assertEqual(expected, func_names)

    def test_different_shaped_accumulations(self):
        a = biggus.NumpyArrayAdapter(np.random.random(2))
        b = biggus.NumpyArrayAdapter(np.zeros((2, 2)))

        c = biggus.mean(b, axis=1)
        d = a - c

        graph = DaskEngine().graph(d)
        func_names = {task[0].__name__ for task in graph.values()
                      if callable(task[0])}
        expected = {'NumpyArrayAdapter\n(2, 2)', 'NumpyArrayAdapter\n(2,)',
                    'mean\n(axis=1)', 'subtract', 'gather'}
        self.assertEqual(expected, func_names)


class Test__make_nodes(unittest.TestCase):
    def setUp(self):
        colon = slice(None)
        masked = True

        self.a = biggus.zeros((3, 2))
        self.b = self.a + self.a

        self.a_nodes = {'chunk shape: (3, 2)\nsource key: [:, :]\n'
                        '\nuuid_0': ('a LazyChunk function', (colon, colon),
                                     self.a, masked)}
        self.b_nodes = {'array ()\n\n(id: uuid_1)': ('a LazyChunk function',
                                                     'uuid_0',
                                                     self.b, masked, )}

        self.grp = DaskGroup(None)

    @staticmethod
    def fake_uuid():
        for i in range(1000):
            yield 'uuid_{}'.format(i)

    @contextmanager
    def patched_make_nodes(self):
        import mock
        u = self.fake_uuid()
        from functools import partial
        new_uuid = partial(next, u)
        with mock.patch('uuid.uuid4', new_uuid):
            yield self.grp._make_nodes

    def assertNodesEqual(self, expected, actual,
                         bullet_dodge=slice(1, None, 2)):
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected.keys():
            exp = expected[key]
            act = actual[key]
            self.assertEqual(exp[bullet_dodge], act[bullet_dodge])

    def test_constant(self):
        graph = {}
        with self.patched_make_nodes() as make_nodes:
            nodes = make_nodes(graph, self.a, list(range(2)), masked=True)
        self.assertNodesEqual(self.a_nodes, nodes)

#     def test_elementwise(self):
#         graph = {}
#         with self.patched_make_nodes() as make_nodes:
#             nodes = make_nodes(graph, self.b, list(range(2)), masked=True)
#         print(nodes)
#         self.assertNodesEqual(self.b_nodes, nodes, slice(1, None))


if __name__ == '__main__':
    unittest.main()
