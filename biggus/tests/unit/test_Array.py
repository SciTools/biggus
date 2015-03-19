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

import unittest

import numpy as np

from biggus import Array


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
        pass

    def ndarray(self, keys):
        pass

    def masked_array(self, keys):
        pass


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
                   "<Array shape=(4,) dtype=dtype('float64') size=32 B>")

    def test_nd(self):
        self._test((2, 6, 5), 'f8',
                   "<Array shape=(2, 6, 5) dtype=dtype('float64') size=480 B>")

    def test_1023(self):
        self._test(1023, 'i1',
                   "<Array shape=(1023,) dtype=dtype('int8') size=1023 B>")

    def test_1024(self):
        self._test(1024, 'i1',
                   "<Array shape=(1024,) dtype=dtype('int8') size=1.00 KiB>")

    def test_40000(self):
        self._test(40000, 'i1',
                   "<Array shape=(40000,) dtype=dtype('int8') size=39.06 KiB>")

    def test_999999(self):
        self._test(
            9999999, 'i1',
            "<Array shape=(9999999,) dtype=dtype('int8') size=9.54 MiB>")

    def test_999999999(self):
        self._test(
            9999999999, 'i1',
            "<Array shape=(9999999999,) dtype=dtype('int8') size=9.31 GiB>")

    def test_999999999999(self):
        self._test(
            9999999999999, 'i1',
            "<Array shape=(9999999999999,) dtype=dtype('int8') size=9.09 TiB>")

    def test_999999999999999(self):
        self._test(
            9999999999999999, 'i1',
            "<Array shape=(9999999999999999,) dtype=dtype('int8') "
            "size=9094.95 TiB>")


class Test___hash__(unittest.TestCase):
    def test_unhashable(self):
        array = FakeArray((3, 4), 'f4')
        with self.assertRaises(TypeError):
            hash(array)


if __name__ == '__main__':
    unittest.main()
