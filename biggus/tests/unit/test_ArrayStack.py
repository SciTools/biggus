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
"""Unit tests for `biggus.ArrayStack`."""

import unittest

import mock
import numpy as np

from biggus import ArrayStack


def fake_array(fill_value):
    return mock.Mock(shape=mock.sentinel.SHAPE, dtype=mock.sentinel.DTYPE,
                     fill_value=fill_value)


class Test___init___fill_values(unittest.TestCase):
    def test_nan_nan(self):
        array1 = fake_array(np.nan)
        array2 = fake_array(np.nan)
        stack = ArrayStack(np.array([array1, array2]))
        self.assertTrue(np.isnan(stack.fill_value))

    def test_nan_number(self):
        array1 = fake_array(np.nan)
        array2 = fake_array(42)
        with self.assertRaises(ValueError):
            stack = ArrayStack(np.array([array1, array2]))

    def test_number_nan(self):
        array1 = fake_array(42)
        array2 = fake_array(np.nan)
        with self.assertRaises(ValueError):
            stack = ArrayStack(np.array([array1, array2]))

    def test_number_number(self):
        array1 = fake_array(42)
        array2 = fake_array(42)
        stack = ArrayStack(np.array([array1, array2]))
        self.assertEqual(stack.fill_value, 42)

    def test_number_other_number(self):
        array1 = fake_array(42)
        array2 = fake_array(43)
        with self.assertRaises(ValueError):
            stack = ArrayStack(np.array([array1, array2]))


if __name__ == '__main__':
    unittest.main()
