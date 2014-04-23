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

import mock
import numpy as np

from biggus import LinearMosaic


def fake_array(fill_value, dtype=np.dtype('f4')):
    return mock.Mock(shape=(3, 4), dtype=dtype, fill_value=fill_value)


class Test___init___fill_values(unittest.TestCase):
    def test_nan_nan(self):
        array1 = fake_array(np.nan)
        array2 = fake_array(np.nan)
        mosaic = LinearMosaic(np.array([array1, array2]), 0)
        self.assertTrue(np.isnan(mosaic.fill_value))

    def test_nan_number(self):
        array1 = fake_array(np.nan)
        array2 = fake_array(42)
        with self.assertRaises(ValueError):
            mosaic = LinearMosaic(np.array([array1, array2]), 0)

    def test_number_nan(self):
        array1 = fake_array(42)
        array2 = fake_array(np.nan)
        with self.assertRaises(ValueError):
            mosaic = LinearMosaic(np.array([array1, array2]), 0)

    def test_number_number(self):
        array1 = fake_array(42)
        array2 = fake_array(42)
        mosaic = LinearMosaic(np.array([array1, array2]), 0)
        self.assertEqual(mosaic.fill_value, 42)

    def test_number_other_number(self):
        array1 = fake_array(42)
        array2 = fake_array(43)
        with self.assertRaises(ValueError):
            mosaic = LinearMosaic(np.array([array1, array2]), 0)

    def test_matching_strings(self):
        array1 = fake_array('foo', np.dtype('S3'))
        array2 = fake_array('foo', np.dtype('S3'))
        mosaic = LinearMosaic(np.array([array1, array2]), 0)
        self.assertEqual(mosaic.fill_value, 'foo')

    def test_different_strings(self):
        array1 = fake_array('foo', np.dtype('S3'))
        array2 = fake_array('bar', np.dtype('S3'))
        with self.assertRaises(ValueError):
            mosaic = LinearMosaic(np.array([array1, array2]), 0)


if __name__ == '__main__':
    unittest.main()
