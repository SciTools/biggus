# (C) British Crown Copyright 2014, Met Office
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
"""Unit tests for `biggus._ArrayAdapter`."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal
from numpy.lib.stride_tricks import as_strided


from biggus import BroadcastArray, NumpyArrayAdapter


class Test___init__(unittest.TestCase):
    def test_array(self):
        orig = np.empty([1, 3, 5])
        a = BroadcastArray(orig, {})
        self.assertIs(a.array, orig)

    def test_invalid_broadcast_axis(self):
        msg = 'Axis -1 out of range \[0, 5\)'
        with self.assertRaisesRegexp(ValueError, msg):
            a = BroadcastArray(np.empty([1, 3, 1, 5, 1]), {-1: 10})

    def test_invalid_broadcast_length(self):
        msg = 'Axis length must be positive. Got -1.'
        with self.assertRaisesRegexp(ValueError, msg):
            a = BroadcastArray(np.empty([1, 3, 1, 5, 1]), {0: -1})

    def test_broadcasting_existing_non_len1_dimension(self):
        msg = 'Attempted to broadcast axis 0 which is of length 3.'
        with self.assertRaisesRegexp(ValueError, msg):
            a = BroadcastArray(np.empty([3]), {0: 5})

    def test_nested_broadcast_avoidance(self):
        orig = np.empty([1, 3, 1, 5, 1])
        a = BroadcastArray(orig, {0: 10, 4: 100})
        b = BroadcastArray(a, {0: 5, 2: 15})
        self.assertIs(b.array, orig)
        self.assertEqual(a._broadcast_dict, {0: 10, 4: 100})
        self.assertEqual(b._broadcast_dict, {0: 5, 2: 15, 4: 100})


class Test_shape(unittest.TestCase):
    def test_broadcast_shape(self):
        a = BroadcastArray(np.empty([1, 3, 1, 5, 1]), {0: 10, 2: 0, 4: 15})
        self.assertEqual(a.shape, (10, 3, 0, 5, 15))


class Test___getitem__(unittest.TestCase):
    def test_nothing_done(self):
        orig = np.empty([1, 3, 1, 5, 1])
        a = BroadcastArray(orig, {0: 10, 2: 0, 4: 15})
        result = a[...]
        self.assertIsInstance(result, BroadcastArray)
        self.assertIs(result.array, orig)
        self.assertEqual(result.shape, (10, 3, 0, 5, 15))

    def test_index_exising_broadcast(self):
        orig = np.empty([1, 3, 1, 5, 1])
        a = BroadcastArray(orig, {0: 10, 2: 0, 4: 15})
        result = a[:-1]
        self.assertIs(result.array, orig)
        self.assertEqual(result.shape, (9, 3, 0, 5, 15))

    def test_index_contained_array_dimension(self):
        orig = np.empty([1, 3, 1, 5, 1])
        a = BroadcastArray(orig, {0: 10, 2: 0, 4: 15})
        result = a[:, -1]
        assert_array_equal(result.array, orig[:, -1])
        self.assertEqual(result.shape, (10, 0, 5, 15))


class Test_ndarray(unittest.TestCase):
    def test_indexed(self):
        orig = np.empty([1, 3, 1, 5, 1], dtype='>i4')
        a = BroadcastArray(orig, {0: 10, 2: 0, 4: 15})
        result = a[:, -1, ::2, ::2]
        expected = as_strided(orig, shape=(10, 0, 3, 15),
                              strides=(0, 0, 4, 0))
        assert_array_equal(result.ndarray(), expected)


class Test_masked_array(unittest.TestCase):
    def test_simple(self):
        orig = np.ma.masked_array([[1], [2], [3]],
                                  mask=[[1], [0], [1]])
        array = BroadcastArray(orig, {1: 2})
        result = array.masked_array()
        expected = np.ma.masked_array([[1, 1], [2, 2], [3, 3]],
                                      mask=[[1, 1], [0, 0], [1, 1]])
        assert_array_equal(result.mask, expected.mask)
        assert_array_equal(result.data, expected.data)

    def test_indexed(self):
        orig = np.ma.masked_array([[1], [2], [3]], mask=[[1], [0], [1]])
        a = BroadcastArray(orig, {1: 2})
        result = a[0:2, :-1].masked_array()
        expected = np.ma.masked_array([[1], [2]], mask=[[1], [0]])
        assert_array_equal(result.mask, expected.mask)
        assert_array_equal(result.data, expected.data)


class Test_broadcast_numpy_array(unittest.TestCase):
    def test_simple_broadcast(self):
        a = np.arange(3, dtype='>i4').reshape([3, 1])
        result = BroadcastArray.broadcast_numpy_array(a, {1: 4})
        expected = np.array([[0, 0, 0, 0],
                             [1, 1, 1, 1],
                             [2, 2, 2, 2]])
        assert_array_equal(result, expected)
        self.assertEqual(result.strides, (4, 0))


if __name__ == '__main__':
    unittest.main()
