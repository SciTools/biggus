# (C) British Crown Copyright 2016, Met Office
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
"""Unit tests for `biggus._init.MaskValueArray`."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from biggus._init import MaskValueArray


class Test___init__(unittest.TestCase):
    def test_nd_array(self):
        orig = np.arange(24)
        array = MaskValueArray(orig, 3)
        self.assertIs(array.array, orig)


class Test___getitem__(unittest.TestCase):
    def test_mask_preserved(self):
        sliced = MaskValueArray(np.arange(5), 3)[2:]
        expected = np.ma.MaskedArray([2, 3, 4], [False, True, False])
        assert_array_equal(sliced.masked_array(), expected)


class Test_ndarray(unittest.TestCase):
    def test_nd_array(self):
        array = MaskValueArray(np.arange(3), 2)
        assert_array_equal(array.ndarray(), np.array([0, 1, 2]))


class Test_masked_array(unittest.TestCase):
    def test_basic(self):
        array = MaskValueArray(np.arange(3), 2)
        expected = np.ma.MaskedArray([0, 1, 2], [False, False, True])
        self.assertIsInstance(array.masked_array(), np.ma.MaskedArray)
        assert_array_equal(array.masked_array(), expected)

    def test_masked_array(self):
        """
        Test to make sure that if the original array is masked, its mask is
        applied by MaskValueArray.masked_array()

        """
        orig = np.ma.MaskedArray(np.arange(3), [True, False, False])
        array = MaskValueArray(orig, 2)
        expected = np.ma.MaskedArray([0, 1, 2], [True, False, True])
        self.assertIsInstance(array.masked_array(), np.ma.MaskedArray)
        assert_array_equal(array.masked_array(), expected)


if __name__ == '__main__':
    unittest.main()
