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

from itertools import permutations
from functools import partial
import unittest

import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal

from biggus import _Elementwise as Elementwise


class Test__masked_arrays(unittest.TestCase):
    def setUp(self):
        a = np.arange(6).reshape(3, 2) - 3
        self.mask = a % 3 == 0
        self.masked_array = np.ma.masked_array(a, self.mask)

    def test_single_argument_operation(self):
        expected = np.abs(self.masked_array)
        actual = Elementwise(self.masked_array, None, np.abs, ma.abs)
        result = actual.masked_array()
        assert_array_equal(result.mask, self.mask)
        assert_array_equal(result, expected)

    def test_dual_argument_operation(self):
        exponent = 2
        expected = self.masked_array ** exponent
        actual = Elementwise(self.masked_array, exponent, np.power, ma.power)
        result = actual.masked_array()
        assert_array_equal(result.mask, self.mask)
        assert_array_equal(result, expected)

    def test_no_masked_array_function(self):
        result = Elementwise(self.masked_array, None, np.sign)
        msg = 'No sign operation defined for masked arrays'
        with self.assertRaisesRegexp(TypeError, msg):
            result.masked_array()


class Test_dual_argument_dtype(unittest.TestCase):
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


class Test_single_argument_dtype(unittest.TestCase):
    def setUp(self):
        dtypes = []
        for dtype in ['int', 'uint', 'float']:
            dtypes.extend(np.sctypes[dtype])
        self.dtypes = dtypes

    def dtype_after_of(self, dtype, op):
        return op(np.ones(1, dtype=dtype)).dtype

    def _test(self, *ops):
        for dtype in self.dtypes:
            a1 = np.ones(1, dtype=dtype)
            actual = Elementwise(a1, None, *ops)
            expected_dtype = self.dtype_after_of(dtype, ops[0])
            self.assertEqual(actual.dtype, expected_dtype)

    def test_abs(self):
        self._test(np.abs, ma.abs)

    def test_rint(self):
        self._test(np.rint)

    def test_sign(self):
        self._test(np.sign)

    def test_square(self):
        self._test(np.square)

    def test_partial_function(self):
        clip_between = partial(np.clip, a_min=1, a_max=4)
        self._test(clip_between)

        array = np.arange(10000, dtype=np.float32)
        actual = Elementwise(array, None, clip_between)
        assert_array_equal(actual.ndarray(), clip_between(array))
        self.assertEqual(actual.ndarray().max(), 4)


class Test_single_argument(unittest.TestCase):
    def test_getitem(self):
        a1 = np.arange(3)
        actual = Elementwise(a1, None, np.cos)
        self.assertEqual(actual[1].ndarray(), np.cos(a1[1]))


if __name__ == '__main__':
    unittest.main()
