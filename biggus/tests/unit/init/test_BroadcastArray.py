# (C) British Crown Copyright 2014 - 2015, Met Office
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


from biggus import BroadcastArray, NumpyArrayAdapter, ConstantArray


class Test___init__(unittest.TestCase):
    def test_array(self):
        orig = np.empty([1, 3, 5])
        a = BroadcastArray(orig, {})
        self.assertIs(a.array, orig)

    def test_invalid_broadcast_axis(self):
        msg = 'Axis -1 out of range \[0, 5\)'
        with self.assertRaisesRegexp(ValueError, msg):
            BroadcastArray(np.empty([1, 3, 1, 5, 1]), {-1: 10})

    def test_invalid_broadcast_length(self):
        msg = 'Axis length must be positive. Got -1.'
        with self.assertRaisesRegexp(ValueError, msg):
            BroadcastArray(np.empty([1, 3, 1, 5, 1]), {0: -1})

    def test_broadcasting_existing_non_len1_dimension(self):
        msg = 'Attempted to broadcast axis 0 which is of length 3.'
        with self.assertRaisesRegexp(ValueError, msg):
            BroadcastArray(np.empty([3]), {0: 5})

    def test_nested_broadcast_avoidance(self):
        orig = np.empty([1, 3, 1, 5, 1])
        a = BroadcastArray(orig, {0: 10, 4: 100}, (3, 2, 1))
        b = BroadcastArray(a, {0: 5, 2: 15}, (4,))
        self.assertIs(b.array, orig)
        self.assertEqual(a._broadcast_dict, {0: 10, 4: 100})
        self.assertEqual(b._broadcast_dict, {0: 5, 2: 15, 4: 100})
        self.assertEqual(b._leading_shape, (4, 3, 2, 1))
        self.assertEqual(b.shape, (4, 3, 2, 1, 5, 3, 15, 5, 100))

    def test_leading_shape(self):
        a = BroadcastArray(np.empty([1, 3]), {0: 2}, (5, 3))
        self.assertEqual(a._shape, (5, 3, 2, 3))

    def test_invalid_leading_shape(self):
        msg = 'Leading shape must all be >=1'
        with self.assertRaisesRegexp(ValueError, msg):
            BroadcastArray(np.empty([1]), {}, (-1,))


class Test_shape(unittest.TestCase):
    def test_broadcast_shape(self):
        a = BroadcastArray(np.empty([1, 3, 1, 5, 1]), {0: 10, 2: 0, 4: 15},
                           leading_shape=(3, 4))
        self.assertEqual(a.shape, (3, 4, 10, 3, 0, 5, 15))


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

    def test_leading_shape_preserve(self):
        orig = np.empty([5, 1])
        a = BroadcastArray(orig, {1: 10}, leading_shape=(3, 4, 2))
        self.assertEqual(a[:, -1].shape, (3, 2, 5, 10))

    def test_remove_leading_and_broadcast(self):
        a = BroadcastArray(np.empty([1, 2, 1]), {0: 3, 2: 3}, (2, 2))
        self.assertEqual(a[:1, 0, 0, 0, :].shape, (1, 3))

    def test_scalar_index_of_broadcast_dimension(self):
        a = BroadcastArray(np.empty((1, 36, 1, 1)), {0: 855, 2: 82, 3: 130})
        self.assertEqual(a[:, :, 30].shape, (855, 36, 130))

    def test_ndarray_indexing(self):
        a = BroadcastArray(np.empty((1, 2)), {0: 5})
        with self.assertRaises(NotImplementedError):
            self.assertEqual(a[np.array([1, 3, 4])].shape, (3, 2))


class Test_ndarray(unittest.TestCase):
    def test_indexed(self):
        orig = np.empty([1, 3, 1, 5, 1], dtype='>i4')
        a = BroadcastArray(orig, {0: 10, 2: 0, 4: 15}, (2,))
        result = a[:, :, -1, ::2, ::2]
        expected = as_strided(orig, shape=(2, 10, 0, 3, 15),
                              strides=(0, 0, 0, 4, 0))
        assert_array_equal(result.ndarray(), expected)


class Test_masked_array(unittest.TestCase):
    def test_simple(self):
        orig = np.ma.masked_array([[1], [2], [3]],
                                  mask=[[1], [0], [1]])
        result = BroadcastArray(orig, {1: 2}, (2,)).masked_array()
        expected, _ = np.broadcast_arrays(orig.data, result)
        expected_mask, _ = np.broadcast_arrays(orig.mask, result)
        expected = np.ma.masked_array(expected, mask=expected_mask)
        assert_array_equal(result.mask, expected.mask)
        assert_array_equal(result.data, expected.data)

    def test_indexed(self):
        orig = np.ma.masked_array([[1], [2], [3]], mask=[[1], [0], [1]])
        a = BroadcastArray(orig, {1: 2})
        result = a[0:2, :-1].masked_array()
        expected = np.ma.masked_array([[1], [2]], mask=[[1], [0]])
        assert_array_equal(result.mask, expected.mask)
        assert_array_equal(result.data, expected.data)


class Test__broadcast_numpy_array(unittest.TestCase):
    def test_simple_broadcast(self):
        a = np.arange(3, dtype='>i4').reshape([3, 1])
        result = BroadcastArray._broadcast_numpy_array(a, {1: 4})
        expected = np.array([[0, 0, 0, 0],
                             [1, 1, 1, 1],
                             [2, 2, 2, 2]])
        assert_array_equal(result, expected)
        self.assertEqual(result.strides, (4, 0))

    def test_broadcast_with_leading(self):
        a = np.arange(3, dtype='>i4').reshape([3, 1])
        result = BroadcastArray._broadcast_numpy_array(a, {1: 4}, (1,))
        expected = np.array([[[0, 0, 0, 0],
                              [1, 1, 1, 1],
                              [2, 2, 2, 2]]])
        self.assertEqual(result.strides, (0, 4, 0))
        assert_array_equal(result, expected)


class Test_compute_broadcast_dicts(unittest.TestCase):
    def test_have_enough_memory(self):
        # Using np.broadcast results in the actual data needing to be
        # realised. The code gets around this by using strides = 0.
        # Pick an array size which isn't realistic to realise in a
        # full array.
        a = ConstantArray([int(10 ** i) for i in range(4, 12)])
        self.assertEqual(BroadcastArray._compute_broadcast_kwargs(a.shape,
                                                                  a.shape)[0],
                         tuple(int(10 ** i) for i in range(4, 12)),
                         )

    def assertBroadcast(self, s1, s2, expected_shape, broadcast_kwargs):
        # Assert that the operations are symmetric.
        r1 = BroadcastArray._compute_broadcast_kwargs(s1, s2)
        r2 = BroadcastArray._compute_broadcast_kwargs(s2, s1)

        self.assertEqual(r1[0], expected_shape)
        self.assertEqual(r1[0], r2[0])
        self.assertEqual(r1[1], r2[2])
        self.assertEqual(r1[2], r2[1])

        self.assertEqual(dict(list(zip(['broadcast', 'leading_shape'],
                                       broadcast_kwargs[0]))),
                         r1[1])
        self.assertEqual(dict(list(zip(['broadcast', 'leading_shape'],
                                       broadcast_kwargs[1]))),
                         r1[2])

        actual = np.broadcast(np.empty(s1), np.empty(s2)).shape
        self.assertEqual(expected_shape, actual)

    def test_scalar_broadcasting(self):
        self.assertBroadcast([1, 2, 3], [], (1, 2, 3),
                             broadcast_kwargs=[[{}, ()], [{}, (1, 2, 3)]])

    def test_scalar_scalar(self):
        self.assertBroadcast([], [], (),
                             broadcast_kwargs=[[{}, ()], [{}, ()]])

    def test_rule1_leading_padding(self):
        self.assertBroadcast((1, 2, 3), [3], (1, 2, 3),
                             [[{}, ()], [{}, (1, 2)]])

    def test_rule2_filling_in_1s(self):
        self.assertBroadcast([1, 2, 3], [3, 1, 3], (3, 2, 3),
                             [[{0: 3}, ()], [{1: 2}, ()]])

    def test_broadcast_with_leading(self):
        self.assertBroadcast([1], [5, 10], (5, 10),
                             [[{0: 10}, (5,)], [{}, ()]])

    def test_rule3_value_error(self):
        msg = ('operands could not be broadcast together with shapes '
               '\(2\,3\) \(1\,2\,4\)')
        with self.assertRaisesRegexp(ValueError, msg):
            BroadcastArray._compute_broadcast_kwargs([2, 3], [1, 2, 4])


class Test_broadcast_arrays(unittest.TestCase):
    def test_rh_broadcast(self):
        a = np.empty([1, 2])
        b = np.empty([2])
        a1, b1 = BroadcastArray.broadcast_arrays(a, b)
        self.assertIs(a1, a)
        self.assertIsInstance(b1, BroadcastArray)
        self.assertEqual(b1.shape, (1, 2))

    def test_lh_broadcast(self):
        a = np.empty([2])
        b = np.empty([1, 2])
        a1, b1 = BroadcastArray.broadcast_arrays(a, b)
        self.assertIs(b1, b)
        self.assertIsInstance(a1, BroadcastArray)
        self.assertEqual(a1.shape, (1, 2))

    def test_both_broadcast(self):
        a = np.empty([1, 2])
        b = np.empty([3, 1])
        a1, b1 = BroadcastArray.broadcast_arrays(a, b)
        self.assertIsInstance(a1, BroadcastArray)
        self.assertIsInstance(b1, BroadcastArray)
        self.assertEqual(a1.shape, (3, 2))
        self.assertEqual(b1.shape, (3, 2))


if __name__ == '__main__':
    unittest.main()
