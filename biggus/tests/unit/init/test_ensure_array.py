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
"""Unit tests for `biggus.ensure_array`."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from biggus import ensure_array, Array, NumpyArrayAdapter
from biggus.tests import mock


class Test_ensure_array(unittest.TestCase):
    def test_array_instance(self):
        array = mock.Mock(spec=Array)
        result = ensure_array(array)
        self.assertIs(array, result)

    def test_numpy_ndarray_instance(self):
        array = np.arange(10)
        result = ensure_array(array)
        self.assertIsInstance(result, NumpyArrayAdapter)
        assert_array_equal(array, result.ndarray())

    def test_other_type(self):
        with self.assertRaises(TypeError):
            ensure_array(range(10))


if __name__ == '__main__':
    unittest.main()
