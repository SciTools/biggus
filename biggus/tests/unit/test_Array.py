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
"""Unit tests for `biggus.Array`."""
from __future__ import division

import sys
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import biggus
from biggus import Array


RESULT_NDARRAY = np.arange(12).reshape(3, 4)


class FakeArray(Array):
    def __init__(self, shape, dtype='f8'):
        self._shape = shape
        self._dtype = np.dtype(dtype)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, keys):
        raise ValueError('getitem called on fake array.')

    def ndarray(self):
        return RESULT_NDARRAY

    def masked_array(self, keys):
        raise ValueError('masked_array called on fake array.')


class Test___array__(unittest.TestCase):
    def test(self):
        array = FakeArray((2, 3))
        result = array.__array__()
        self.assertIs(result, RESULT_NDARRAY)

    def test_dtype(self):
        array = FakeArray((2, 3))
        result = array.__array__('i4')
        assert_array_equal(result, RESULT_NDARRAY)
        self.assertEqual(result.dtype, np.dtype('i4'))


class Test_nbytes(unittest.TestCase):
    def _test(self, shape, dtype=np.dtype('f8')):
        array = FakeArray(shape, dtype)
        ndarray = np.empty(shape, dtype)
        self.assertEqual(array.nbytes, ndarray.nbytes)

    def test_type(self):
        self.assertIsInstance(FakeArray((3, 4)).nbytes, int)

    def test_0d(self):
        self._test(())

    def test_1d(self):
        self._test(4)

    def test_nd(self):
        self._test((2, 6, 1, 5))

    def test_dtype(self):
        self._test((2, 6, 1, 5), 'i2')


class Test___str__(unittest.TestCase):
    def _test(self, shape, dtype, expected):
        if not isinstance(shape, tuple):
            shape = (shape,)
        array = FakeArray(shape, dtype)
        self.assertEqual(str(array), expected)

    def test_0d(self):
        self._test((), 'f8',
                   "<Array shape=() dtype=dtype('float64') size=8 B>")

    def test_1d(self):
        self._test(4, 'f8',
                   "<Array shape=(4) dtype=dtype('float64') size=32 B>")

    def test_nd(self):
        self._test((2, 6, 5), 'f8',
                   "<Array shape=(2, 6, 5) dtype=dtype('float64') size=480 B>")

    def test_1023(self):
        self._test(1023, 'i1',
                   "<Array shape=(1023) dtype=dtype('int8') size=1023 B>")

    def test_1024(self):
        self._test(1024, 'i1',
                   "<Array shape=(1024) dtype=dtype('int8') size=1.00 KiB>")

    def test_40000(self):
        self._test(40000, 'i1',
                   "<Array shape=(40000) dtype=dtype('int8') size=39.06 KiB>")

    def test_999999(self):
        self._test(
            9999999, 'i1',
            "<Array shape=(9999999) dtype=dtype('int8') size=9.54 MiB>")

    def test_999999999(self):
        self._test(
            9999999999, 'i1',
            "<Array shape=(9999999999) dtype=dtype('int8') size=9.31 GiB>")

    def test_999999999999(self):
        self._test(
            9999999999999, 'i1',
            "<Array shape=(9999999999999) dtype=dtype('int8') size=9.09 TiB>")

    def test_999999999999999(self):
        self._test(
            9999999999999999, 'i1',
            "<Array shape=(9999999999999999) dtype=dtype('int8') "
            "size=9094.95 TiB>")


class Test___hash__(unittest.TestCase):
    def test_unhashable(self):
        array = FakeArray((3, 4), 'f4')
        with self.assertRaises(TypeError):
            hash(array)


class Test_astype(unittest.TestCase):
    def test(self):
        array = FakeArray((2, 3, 4))
        result = array.astype('>f32')
        self.assertIsInstance(result, biggus.AsDataTypeArray)
        self.assertEqual(result.dtype, '>f32')


class Test_transpose(unittest.TestCase):
    def test_default(self):
        array = FakeArray((2, 3, 4))
        result = array.transpose()
        self.assertIsInstance(result, biggus.TransposedArray)
        self.assertEqual(tuple(result.axes), (2, 1, 0))

    def test_explicit(self):
        array = FakeArray((2, 3, 4))
        result = array.transpose((1, 2, 0))
        self.assertIsInstance(result, biggus.TransposedArray)
        self.assertEqual(result.axes, (1, 2, 0))


class AssertElementwiseMixin(object):
    def assertElementwise(self, ag1, ag2):
        self.assertIs(ag1._array1, ag2._array1)
        self.assertIs(ag1._array2, ag2._array2)
        self.assertIs(ag1._numpy_op, ag2._numpy_op)
        self.assertIs(ag1._ma_op, ag2._ma_op)


class Test___add__(unittest.TestCase, AssertElementwiseMixin):
    def test_other_array(self):
        a = FakeArray([2, 4])
        b = FakeArray([2, 4])
        r = a + b
        self.assertIsInstance(r, biggus._Elementwise)
        self.assertElementwise(r, biggus.add(a, b))

    def test_other_no_good(self):
        a = FakeArray([2, 2])
        with self.assertRaisesRegexp(TypeError, 'unsupported operand type'):
            a + None

    def test___radd__(self):
        a = biggus.NumpyArrayAdapter(RESULT_NDARRAY)
        r = 5 + a
        self.assertIsInstance(r, biggus._Elementwise)
        assert_array_equal(r.ndarray(), 5 + RESULT_NDARRAY)


class Test___sub__(unittest.TestCase, AssertElementwiseMixin):
    def test_other_array(self):
        a = FakeArray([2, 4])
        b = FakeArray([2, 4])
        r = a - b
        self.assertIsInstance(r, biggus._Elementwise)
        self.assertElementwise(r, biggus.sub(a, b))

    def test_other_no_good(self):
        a = FakeArray([2, 2])
        with self.assertRaisesRegexp(TypeError, 'unsupported operand type'):
            a - None

    def test___rsub__(self):
        a = biggus.NumpyArrayAdapter(RESULT_NDARRAY)
        r = 5 - a
        self.assertIsInstance(r, biggus._Elementwise)
        assert_array_equal(r.ndarray(), 5 - RESULT_NDARRAY)


class Test___mul__(unittest.TestCase, AssertElementwiseMixin):
    def test_other_array(self):
        a = FakeArray([2, 4])
        b = FakeArray([2, 4])
        r = a * b
        self.assertIsInstance(r, biggus._Elementwise)
        self.assertElementwise(r, biggus.multiply(a, b))

    def test_other_no_good(self):
        a = FakeArray([2, 2])
        with self.assertRaisesRegexp(TypeError, 'unsupported operand type'):
            a * None

    def test___rmul__(self):
        a = biggus.NumpyArrayAdapter(RESULT_NDARRAY)
        r = 5 * a
        self.assertIsInstance(r, biggus._Elementwise)
        assert_array_equal(r.ndarray(), 5 * RESULT_NDARRAY)


class Test___floordiv__(unittest.TestCase, AssertElementwiseMixin):
    def test_other_array(self):
        a = FakeArray([2, 4])
        b = FakeArray([2, 4])
        r = a // b
        self.assertIsInstance(r, biggus._Elementwise)
        self.assertElementwise(r, biggus.floor_divide(a, b))

    def test_other_no_good(self):
        a = FakeArray([2, 2])
        with self.assertRaisesRegexp(TypeError, 'unsupported operand type'):
            a // None

    def test___rfloordiv__(self):
        a = biggus.NumpyArrayAdapter(RESULT_NDARRAY + 10)
        r = 5 // a
        self.assertIsInstance(r, biggus._Elementwise)
        assert_array_equal(r.ndarray(), 5 // (RESULT_NDARRAY + 10))


class Test___div__(unittest.TestCase, AssertElementwiseMixin):
    def test_other_array(self):
        a = FakeArray([2, 4])
        b = FakeArray([2, 4])
        r = a.__div__(b)
        self.assertIsInstance(r, biggus._Elementwise)
        self.assertElementwise(r, biggus.divide(a, b))

    def test_other_no_good(self):
        a = FakeArray([2, 2])
        self.assertIs(a.__div__(None), NotImplemented)

    def test___rdiv__(self):
        # We only have rdiv on py2.
        if sys.version_info[0] == 2:
            a = biggus.NumpyArrayAdapter(RESULT_NDARRAY + 10)
            r = 5 / a
            self.assertIsInstance(r, biggus._Elementwise)
            assert_array_equal(r.ndarray(), 5 / (RESULT_NDARRAY + 10))


class Test___trudiv__(unittest.TestCase, AssertElementwiseMixin):
    def test_other_array(self):
        a = FakeArray([2, 4])
        b = FakeArray([2, 4])
        r = a / b
        self.assertIsInstance(r, biggus._Elementwise)
        self.assertElementwise(r, biggus.true_divide(a, b))

    def test_other_no_good(self):
        a = FakeArray([2, 2])
        with self.assertRaisesRegexp(TypeError, 'unsupported operand type'):
            a / None

    def test___rtrudiv__(self):
        a = biggus.NumpyArrayAdapter(RESULT_NDARRAY + 10)
        r = 5 / a
        self.assertIsInstance(r, biggus._Elementwise)
        assert_array_equal(r.ndarray(), 5 / (RESULT_NDARRAY + 10))


class Test___pow__(unittest.TestCase, AssertElementwiseMixin):
    def test_other_array(self):
        a = FakeArray([2, 4])
        b = FakeArray([2, 4])
        r = a ** b
        self.assertIsInstance(r, biggus._Elementwise)
        self.assertElementwise(r, biggus.power(a, b))

    def test_other_no_good(self):
        a = FakeArray([2, 2])
        with self.assertRaisesRegexp(TypeError, 'unsupported operand type'):
            a ** None

    def test___rpow__(self):
        a = biggus.NumpyArrayAdapter(RESULT_NDARRAY)
        r = 5 ** a
        self.assertIsInstance(r, biggus._Elementwise)
        assert_array_equal(r.ndarray(), 5 ** RESULT_NDARRAY)


if __name__ == '__main__':
    unittest.main()
