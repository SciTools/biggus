# (C) British Crown Copyright 2015 - 2016, Met Office
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
"""Unit tests for `biggus._init.AsDataTypeArray`."""

from __future__ import absolute_import, division, print_function
from six.moves import (filter, input, map, range, zip)  # noqa

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from biggus._init import ArrayContainer, NumpyArrayAdapter, AsDataTypeArray


class Test___init__(unittest.TestCase):
    def test_nd_array(self):
        orig = np.arange(24)
        array = AsDataTypeArray(orig, '>f4')
        self.assertEqual(array.dtype, '>f4')
        self.assertIs(array.array, orig)


class Test___getitem__(unittest.TestCase):
    def test_dtype_preserved(self):
        sliced = NumpyArrayAdapter(np.arange(4)).astype('>f4')[:2]
        self.assertEqual(sliced.ndarray().dtype, '>f4')


class Test_ndarray(unittest.TestCase):
    def test_nd_array(self):
        orig = np.arange(3)
        array = AsDataTypeArray(orig, '>f4')
        assert_array_equal(array.ndarray(), np.array([0, 1, 2], dtype='>f4'))


class Test_masked_array(unittest.TestCase):
    def test_nd_array(self):
        orig = np.arange(3)
        array = AsDataTypeArray(orig, '>f4')
        self.assertIsInstance(array.masked_array(), np.ma.masked_array)
        assert_array_equal(array.masked_array(),
                           np.array([0, 1, 2], dtype='>f4'))


if __name__ == '__main__':
    unittest.main()
