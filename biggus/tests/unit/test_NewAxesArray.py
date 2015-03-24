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
"""Unit tests for `biggus.NewAxesArray`."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from biggus import NewAxesArray, ConstantArray, NumpyArrayAdapter


class Test___init__(unittest.TestCase):
    def test_too_few_axes(self):
        in_arr = ConstantArray([3, 2])
        msg = 'must have length 3 but was actually length 2'
        with self.assertRaisesRegexp(ValueError, msg):
            array = NewAxesArray(in_arr, [1, 2])

    def test_too_many_axes(self):
        in_arr = ConstantArray([3, 2])
        msg = 'must have length 3 but was actually length 4'
        with self.assertRaisesRegexp(ValueError, msg):
            array = NewAxesArray(in_arr, [1, 2, 0, 1])

    def test_new_axes_wrong_dtype(self):
        in_arr = ConstantArray([3, 2])
        msg = 'Only positive integer types may be used for new_axes.'
        with self.assertRaisesRegexp(ValueError, msg):
            array = NewAxesArray(in_arr, [1.1, 1, 1])

    def test_new_axes_negative(self):
        in_arr = ConstantArray([1, 2])
        msg = 'Only positive integer types may be used for new_axes.'
        with self.assertRaisesRegexp(ValueError, msg):
            array = NewAxesArray(in_arr, [-1, 0, 1])

    def test_successful_attribute(self):
        in_arr = ConstantArray([3, 1])
        array = NewAxesArray(in_arr, [2, 0, 4])
        self.assertIs(array.array, in_arr)
        self.assertIsInstance(array._new_axes, np.ndarray)
        self.assertEqual(list(array._new_axes), [2, 0, 4])

    def test_0d(self):
        array = NewAxesArray(np.array(0), [3])
        self.assertEqual(list(array._new_axes), [3])

    def test_nested_newaxes(self):
        # Currently it is possible to nest NewAxesArrays. It would be
        # nice if this weren't the case.
        orig = np.empty([3])
        array1 = NewAxesArray(orig, [1, 0])
        array2 = NewAxesArray(array1, [0, 0, 2])
        self.assertEqual(array2.shape, (1, 3, 1, 1))
        self.assertIs(array2.array, array1)


class Test_shape(unittest.TestCase):
    def test_no_new_axes(self):
        in_arr = ConstantArray([3, 1])
        array = NewAxesArray(in_arr, [0, 0, 0])
        self.assertEqual(array.shape, (3, 1))

    def test_many_new_axes(self):
        in_arr = ConstantArray([3, 2])
        array = NewAxesArray(in_arr, [3, 1, 2])
        self.assertEqual(array.shape, (1, 1, 1, 3, 1, 2, 1, 1))

    def test_left_only_new_axes(self):
        in_arr = ConstantArray([3, 2])
        array = NewAxesArray(in_arr, [1, 0, 0])
        self.assertEqual(array.shape, (1, 3, 2))

    def test_right_only_new_axes(self):
        in_arr = ConstantArray([3, 2])
        array = NewAxesArray(in_arr, [0, 0, 2])
        self.assertEqual(array.shape, (3, 2, 1, 1))

    def test_0d(self):
        array = NewAxesArray(np.array(0), [3])
        self.assertEqual(array.shape, (1, 1, 1))


# TODO: Remove once my other PR is in.
class _KeyGen(object):
    def __getitem__(self, keys):
        return keys
key_gen = _KeyGen()


class Test__newaxis_keys(unittest.TestCase):
    def assert_newaxis_keys(self, shape, new_axes, expected):
        in_arr = ConstantArray(shape)
        array = NewAxesArray(in_arr, new_axes)
        keys = array._newaxis_keys()
        self.assertEqual(keys, expected)

    def test_no_new_axes(self):
        self.assert_newaxis_keys([3, 2], [0, 0, 0], key_gen[:, :])

    def test_one_new_axes(self):
        new_ax = np.newaxis
        self.assert_newaxis_keys([3, 2], [1, 1, 1],
                                 key_gen[new_ax, :, new_ax, :, new_ax])

    def test_many_new_axes(self):
        new_ax = np.newaxis
        self.assert_newaxis_keys([3, 2], [3, 2, 2],
                                 key_gen[new_ax, new_ax, new_ax, :,
                                         new_ax, new_ax, :,
                                         new_ax, new_ax])


class Test___getitem__(unittest.TestCase):
    def setUp(self):
        self.array = NewAxesArray(np.arange(24).reshape(4, 3, 2),
                                  [1, 2, 0, 1])
        self.array_3d = NewAxesArray(np.arange(3), [1, 1])

    def test_new_axis_ellipsis_leading(self):
        result = self.array[..., np.newaxis]
        self.assertIsInstance(result, NewAxesArray)
        self.assertEqual(list(result._new_axes), [1, 2, 0, 2])

    def test_new_axis_ellipsis_trailing(self):
        result = self.array[np.newaxis, ...]
        self.assertIsInstance(result, NewAxesArray)
        self.assertEqual(list(result._new_axes), [2, 2, 0, 1])

    def test_new_axis_with_combine(self):
        result = self.array[np.newaxis, :, 0]
        self.assertIsInstance(result, NewAxesArray)
        self.assertEqual(list(result._new_axes), [4, 0, 1])

    def test_index_existing_newaxis(self):
        result = self.array[0, :, 0]
        self.assertIsInstance(result, NewAxesArray)
        self.assertEqual(list(result._new_axes), [0, 1, 0, 1])

    def test_index_existing_newaxis(self):
        result = self.array[0, 0, 0, ..., 0]
        self.assertIsInstance(result, NewAxesArray)
        self.assertEqual(list(result._new_axes), [1, 0, 0])

    def test_index_existing_newaxis(self):
        result = self.array[0, 0, 0, 0, 0, 0, 0]
        self.assertIsInstance(result, NewAxesArray)
        self.assertEqual(list(result._new_axes), [0])

    def test_combine_all_with_newaxis(self):
        result = self.array[0, 0, np.newaxis, 0, 0, np.newaxis, 0, 0, 0]
        self.assertIsInstance(result, NewAxesArray)
        self.assertEqual(list(result._new_axes), [2])

    def test_new_axis_valid_slice(self):
        self.assertEqual(self.array_3d[0:1, ..., 0:1].shape, (1, 3, 1))

    def test_new_axis_invalid_slice(self):
        self.assertEqual(self.array_3d[1:2, ..., 3:1].shape, (0, 3, 0))

    def test_new_axis_valid_index(self):
        self.assertEqual(self.array_3d[0, ..., 0].shape, (3, ))

    def test_new_axis_invalid_index(self):
        with self.assertRaises(IndexError):
            self.array_3d[1]

        with self.assertRaises(IndexError):
            self.array_3d[-2]

    def test_new_axis_tuple_indexing(self):
        self.assertEqual(self.array_3d[(0, 0, 0), ...].shape, (3, 3, 1))

    def test_new_axis_numpy_array_indexing(self):
        msg = "NewAxesArray indexing not yet supported for ndarray keys."
        with self.assertRaisesRegexp(NotImplementedError, msg):
            self.array_3d[np.array([0, 0, 0]), ...]


class Test_ndarray(unittest.TestCase):
    def test_numpy_array(self):
        array = NewAxesArray(np.arange(24), [1, 2])
        self.assertIsInstance(array.ndarray(), np.ndarray)
        assert_array_equal(array.masked_array(),
                           np.arange(24).reshape(1, 24, 1, 1))


class Test_masked_array(unittest.TestCase):
    def test_numpy_array(self):
        array = NewAxesArray(np.arange(24), [1, 2])
        self.assertIsInstance(array.masked_array(), np.ma.MaskedArray)
        assert_array_equal(array.masked_array(),
                           np.arange(24).reshape(1, 24, 1, 1))


if __name__ == '__main__':
    unittest.main()
