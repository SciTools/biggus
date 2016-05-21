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
"""Unit tests for `biggus.LinearMosaic`."""

import unittest

import numpy as np

from biggus import Array, LinearMosaic, ConstantArray
from biggus.tests import mock


class Test___init___invalid(unittest.TestCase):
    def test_not_arrays(self):
        class BadArray(object):
            dtype = 'f'
            fill_value = 9
            ndim = 1
            shape = (4,)
        bad_arrays = [BadArray()]
        with self.assertRaisesRegexp(ValueError, 'subclass'):
            LinearMosaic(bad_arrays, 0)


def fake_array(fill_value, dtype=np.dtype('f4')):
    return mock.Mock(shape=(3, 4), dtype=dtype, fill_value=fill_value,
                     ndim=2, spec=Array)


class Test___init___fill_values(unittest.TestCase):
    def test_nan_nan(self):
        array1 = fake_array(np.nan)
        array2 = fake_array(np.nan)
        mosaic = LinearMosaic(np.array([array1, array2]), 0)
        self.assertTrue(np.isnan(mosaic.fill_value))

    def test_nan_number(self):
        array1 = fake_array(np.nan)
        array2 = fake_array(42)
        mosaic = LinearMosaic(np.array([array1, array2]), 0)
        self.assertEqual(mosaic.fill_value, 1e+20)

    def test_number_nan(self):
        array1 = fake_array(42)
        array2 = fake_array(np.nan)
        mosaic = LinearMosaic(np.array([array1, array2]), 0)
        self.assertEqual(mosaic.fill_value, 1e+20)

    def test_number_number(self):
        array1 = fake_array(42)
        array2 = fake_array(42)
        mosaic = LinearMosaic(np.array([array1, array2]), 0)
        self.assertEqual(mosaic.fill_value, 42)

    def test_number_other_number(self):
        array1 = fake_array(42)
        array2 = fake_array(43)
        mosaic = LinearMosaic(np.array([array1, array2]), 0)
        self.assertEqual(mosaic.fill_value, 1e+20)

    def test_matching_strings(self):
        array1 = fake_array('foo', np.dtype('S3'))
        array2 = fake_array('foo', np.dtype('S3'))
        mosaic = LinearMosaic(np.array([array1, array2]), 0)
        self.assertEqual(mosaic.fill_value, 'foo')

    def test_different_strings(self):
        array1 = fake_array('foo', np.dtype('S3'))
        array2 = fake_array('bar', np.dtype('S3'))
        mosaic = LinearMosaic(np.array([array1, array2]), 0)
        self.assertEqual(mosaic.fill_value, 'N/A')


class Test___getitem__(unittest.TestCase):
    # Note, these are not a complete set of unit tests.
    # Currently they only handle the newaxis checking.
    # There are more tests in biggus.tests.test_linear_mosaic.
    def setUp(self):
        self.a1 = ConstantArray([2, 3])
        self.a2 = ConstantArray([4, 3])
        self.a = LinearMosaic([self.a1, self.a2], axis=0)

    def test_newaxis_leading(self):
        self.assertEqual(self.a[np.newaxis].shape, (1, 6, 3))

    def test_newaxis_trailing(self):
        self.assertEqual(self.a[..., np.newaxis].shape, (6, 3, 1))


if __name__ == '__main__':
    unittest.main()
