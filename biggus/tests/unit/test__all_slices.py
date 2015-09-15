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
"""Unit tests for `biggus._all_slices`."""

import unittest

import numpy as np

import biggus
from biggus import _all_slices
from biggus.tests import mock


class Test__all_slices(unittest.TestCase):
    def _small_array(self):
        shape = (5, 928, 744)
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        array = biggus.NumpyArrayAdapter(data)
        return array

    def test_min(self):
        array = self._small_array()
        slices = _all_slices(array)
        expected = [[slice(0, 3, None), slice(3, 5, None)],
                    (slice(None, None, None),), (slice(None, None, None),)]
        self.assertEqual(slices, expected)

if __name__ == '__main__':
    unittest.main()
