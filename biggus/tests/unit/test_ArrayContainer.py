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
"""Unit tests for `biggus.ArrayContainer`."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from biggus import ArrayContainer, NumpyArrayAdapter


class Test_ndarray(unittest.TestCase):
    def test_biggus_array(self):
        array = ArrayContainer(NumpyArrayAdapter(np.arange(24)))
        self.assertIsInstance(array.ndarray(), np.ndarray)
        assert_array_equal(array.ndarray(),
                           np.arange(24))

    def test_numpy_array(self):
        array = ArrayContainer(np.arange(24))
        self.assertIsInstance(array.ndarray(), np.ndarray)
        assert_array_equal(array.ndarray(),
                           np.arange(24))

    def test_getitem(self):
        orig = NumpyArrayAdapter(np.arange(4))
        r = ArrayContainer(orig)[:2]
        assert_array_equal(r.ndarray(), orig[:2])


class Test_masked_array(unittest.TestCase):
    def test_biggus_array(self):
        array = ArrayContainer(NumpyArrayAdapter(np.arange(24)))
        self.assertIsInstance(array.masked_array(), np.ma.MaskedArray)
        assert_array_equal(array.masked_array(), np.arange(24))

    def test_numpy_array(self):
        array = ArrayContainer(np.arange(24))
        self.assertIsInstance(array.masked_array(), np.ma.MaskedArray)
        assert_array_equal(array.masked_array(),
                           np.arange(24))


if __name__ == '__main__':
    unittest.main()
