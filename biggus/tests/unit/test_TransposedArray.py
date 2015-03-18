# (C) British Crown Copyright 2015, Met Office
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
"""Unit tests for `biggus.TransposedArray`."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from biggus import TransposedArray, ConstantArray


class Test__apply_axes_mapping(unittest.TestCase):
    def setUp(self):
        self.zeros = ConstantArray([10, 20, 40, 50])
        self.a = TransposedArray(self.zeros, [1, 3, 0, 2])

    def test_too_few_dims(self):
        msg = 'length 3, but should be of length 4'
        with self.assertRaisesRegexp(ValueError, msg):
            self.a._apply_axes_mapping(list('abc'))

    def test_correct_n_dims(self):
        r = self.a._apply_axes_mapping(list('abcd'))
        self.assertEqual(r, ('b', 'd', 'a', 'c'))

    def test_inverse(self):
        r = self.a._apply_axes_mapping(list('bdac'), inverse=True)
        self.assertEqual(r, tuple('abcd'))

    def test_too_many_dims(self):
        msg = 'length 5, but should be of length 4'
        with self.assertRaisesRegexp(ValueError, msg):
            self.a._apply_axes_mapping(list('abcde'))


class Test_shape(unittest.TestCase):
    def transposed_shape(self, shape, axes):
        return TransposedArray(ConstantArray(shape), axes).shape

    def test_no_transpose(self):
        self.assertEqual(self.transposed_shape([10, 20, 30], [0, 1, 2]),
                         (10, 20, 30))

    def test_reverse(self):
        self.assertEqual(self.transposed_shape([10, 20, 30], None),
                         (30, 20, 10))

    def test_non_simple(self):
        self.assertEqual(self.transposed_shape([10, 20, 30], [2, 0, 1]),
                         (30, 10, 20))


class Test___repr__(unittest.TestCase):
    def test_repr(self):
        self.zeros = ConstantArray([10, 20, 40, 50])
        self.a = TransposedArray(self.zeros, [1, 3, 0, 2])

        expected = ("TransposedArray(<ConstantArray shape=(10, 20, 40, 50) "
                    "dtype=dtype('float64')>, [1, 3, 0, 2])")
        self.assertEqual(repr(self.a), expected)


class Test___getitem__(unittest.TestCase):
    def setUp(self):
        self.orig_array = np.empty([5, 6, 4, 7])
        self.arr_transposed = self.orig_array.transpose([1, 3, 0, 2])
        self.a = TransposedArray(self.orig_array, [1, 3, 0, 2])
        self.expected_shape = (6, 7, 5, 4)

    def test_full_slice_indexing(self):
        result = self.a[:, :, :, :]
        self.assertEqual(result.shape, self.expected_shape)
        self.assertIsInstance(result, TransposedArray)
        assert_array_equal(result.ndarray(), self.arr_transposed)

    def test_simple_indexing(self):
        result = self.a[:10, ::2, ::-1, 2:3]
        expected = self.arr_transposed[:10, ::2, ::-1, 2:3]
        self.assertEqual(result.shape, (6, 4, 5, 1))
        assert_array_equal(result.ndarray(), expected)

    def test_dimension_losing_indexing(self):
        result = self.a[:10, 2, ::-1, 1]
        expected = self.arr_transposed[:10, 2, ::-1, 1]
        self.assertEqual(result.shape, (6, 5))
        assert_array_equal(result.ndarray(), expected)

    def test_partial_indexing(self):
        result = self.a[:2, 0]
        expected = self.arr_transposed[:2, 0]
        self.assertEqual(result.shape, (2, 5, 4))
        assert_array_equal(result.ndarray(), expected)

    def test_ellipsis_indexing(self):
        result = self.a[..., :2, 0]
        expected = self.arr_transposed[..., :2, 0]
        self.assertEqual(result.shape, (6, 7, 2))
        assert_array_equal(result.ndarray(), expected)

    def test_overspecified_indexing(self):
        with self.assertRaises(IndexError):
            result = self.a[:2, 0, :, 2, 1]

    def test_new_axis_indexing_more_than_n(self):
        result = self.a[:2, 0, :, np.newaxis, :]
        expected = self.arr_transposed[:2, 0, :, np.newaxis, :]
        self.assertEqual(result.shape, (2, 5, 1, 4))
        assert_array_equal(result.ndarray(), expected)

    def test_axes_non_list(self):
        arr = TransposedArray(self.orig_array, tuple(self.a.axes))
        result = arr[:2, 0, :, np.newaxis, :]
        self.assertEqual(result.shape, (2, 5, 1, 4))


if __name__ == '__main__':
    unittest.main()
