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
"""Unit tests for `biggus._Elementwise`."""

import unittest

from itertools import permutations

import numpy as np
import numpy.ma as ma

from biggus import _Elementwise as Elementwise


class Test_dtype(unittest.TestCase):
    def setUp(self):
        dtypes = []
        for dtype in ['int', 'uint', 'float']:
            dtypes.extend(np.sctypes[dtype])
        self.pairs = [pair for pair in permutations(dtypes, 2)]

    def common_dtype(self, dtype1, dtype2, op):
        return op(np.ones(1, dtype=dtype1),
                  np.ones(1, dtype=dtype2)).dtype

    def _test(self, *ops):
        for dtype1, dtype2 in self.pairs:
            a1 = np.ones(1, dtype=dtype1)
            a2 = np.ones(1, dtype=dtype2)
            actual = Elementwise(a1, a2, *ops)
            expected_dtype = self.common_dtype(dtype1, dtype2, ops[0])
            self.assertEqual(actual.dtype, expected_dtype)

    def test_add(self):
        self._test(np.add, ma.add)

    def test_sub(self):
        self._test(np.subtract, ma.subtract)

    def test_multiply(self):
        self._test(np.multiply, ma.multiply)

    def test_floor_divide(self):
        self._test(np.floor_divide, ma.floor_divide)

    def test_true_divide(self):
        self._test(np.true_divide, ma.true_divide)

    def test_power(self):
        self._test(np.power, ma.power)


if __name__ == '__main__':
    unittest.main()
