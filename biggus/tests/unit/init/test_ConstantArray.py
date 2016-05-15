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
"""Unit tests for `biggus.ConstantArray`."""

import unittest

import numpy as np

from biggus import ConstantArray
from biggus.tests import mock


class Test___init__(unittest.TestCase):
    def test_shape_scalar(self):
        shape = 5
        array = ConstantArray(shape)
        self.assertEqual(array.shape, (shape,))

    def test_shape_tuple(self):
        shape = (30, 10, 20)
        array = ConstantArray(shape)
        self.assertEqual(array.shape, shape)

    def test_shape_tuple(self):
        shape = (30, 10, 20)
        array = ConstantArray(shape)
        self.assertEqual(array.shape, shape)

    def test_shape_scalar_like_int(self):
        array = ConstantArray('34')
        self.assertEqual(array.shape, (34,))

    def test_shape_tuple_like_int(self):
        array = ConstantArray(('34', '93'))
        self.assertEqual(array.shape, (34, 93))

    def test_shape_invalid_scalar(self):
        with self.assertRaises(ValueError):
            array = ConstantArray('foo')

    def test_shape_invalid_tuple(self):
        with self.assertRaises(ValueError):
            array = ConstantArray(('foo', 'bar'))

    def test_value(self):
        value = 6
        array = ConstantArray(3, value)
        self.assertEqual(array.value, value)

    def test_dtype(self):
        dtype = 'i2'
        array = ConstantArray((), dtype=dtype)
        self.assertIs(array.dtype, np.dtype(dtype))

    def test_dtype_default(self):
        array = ConstantArray(())
        self.assertEqual(array.dtype, np.dtype('f8'))

    def test_dtype_default_integer(self):
        array = ConstantArray((), 42)
        self.assertEqual(array.dtype, np.dtype(np.int_))


class Test___getitem__(unittest.TestCase):
    def test_indexing_slice(self):
        shape = (30, 10, 20)
        array = ConstantArray(shape)
        result = array[:5]
        data = np.zeros(shape)[:5]
        self.assertEqual(result.shape, data.shape)

    def test_newaxis(self):
        array = ConstantArray([2, 3])
        result = array[:5, np.newaxis]
        self.assertEqual(result.shape, (2, 1, 3))


class Test_masked_array(unittest.TestCase):
    def test_masked(self):
        shape = (3, 4, 5)
        array = ConstantArray(shape)
        result = array.masked_array()
        self.assertTrue(np.ma.isMaskedArray(result))

    def test_values(self):
        shape = (3, 4, 5)
        array = ConstantArray(shape)
        result = array.masked_array()
        np.testing.assert_array_equal(result, np.ma.zeros(shape))

    def test_dtype(self):
        shape = (3, 4, 5)
        array = ConstantArray(shape, dtype='i4')
        result = array.masked_array()
        self.assertEqual(result.dtype, np.dtype('i4'))

    def test_dtype_default(self):
        shape = (3, 4, 5)
        array = ConstantArray(shape)
        result = array.masked_array()
        self.assertEqual(result.dtype, np.dtype('f8'))


class Test_ndarray(unittest.TestCase):
    def test_values(self):
        shape = (3, 4, 5)
        value = 81
        array = ConstantArray(shape, value)
        result = array.ndarray()
        expected = np.empty(shape)
        expected[()] = value
        np.testing.assert_array_equal(result, expected)

    def test_values_default(self):
        shape = (3, 4, 5)
        array = ConstantArray(shape)
        result = array.ndarray()
        np.testing.assert_array_equal(result, np.zeros(shape))

    def test_dtype(self):
        shape = (3, 4, 5)
        array = ConstantArray(shape, dtype='i4')
        result = array.ndarray()
        self.assertEqual(result.dtype, np.dtype('i4'))

    def test_dtype_default(self):
        shape = (3, 4, 5)
        array = ConstantArray(shape)
        result = array.ndarray()
        self.assertEqual(result.dtype, np.dtype('f8'))


if __name__ == '__main__':
    unittest.main()
