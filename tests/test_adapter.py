# (C) British Crown Copyright 2012, Met Office
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
import unittest

import numpy

import biggus


class TestAdapter(unittest.TestCase):
    def test_dtype(self):
        dtypes = ['f4', 'i1', 'O', 'm8', '<f4', '>f4', '=f4']
        keys = [None, (), (5,), (slice(1, 3),)]
        for dtype in dtypes:
            for key in keys:
                ndarray = numpy.zeros(10, dtype=dtype)
                array = biggus.ArrayAdapter(ndarray, keys=key)
                self.assertEqual(array.dtype, numpy.dtype(dtype))

    def test_shape_0d(self):
        pairs = [
            [None, ()],
        ]
        for key, shape in pairs:
            ndarray = numpy.zeros(())
            array = biggus.ArrayAdapter(ndarray, keys=key)
            self.assertEqual(array.shape, shape)

    def test_shape_1d(self):
        pairs = [
            [None, (10,)],
            [(), (10,)],
            [(5,), ()],
            [(slice(1, 3),), (2,)],
        ]
        for key, shape in pairs:
            ndarray = numpy.zeros(10)
            array = biggus.ArrayAdapter(ndarray, keys=key)
            self.assertEqual(array.shape, shape)

    def test_shape_2d(self):
        pairs = [
            [None, (30, 40)],
            [(), (30, 40)],
            [(5,), (40,)],
            [(slice(1, 3),), (2, 40)],
            [(slice(None, None),), (30, 40)],
            [(5, 3), ()],
            [(5, slice(2, 6)), (4,)],
            [(slice(2, 3), slice(2, 6)), (1, 4)],
        ]
        for key, shape in pairs:
            ndarray = numpy.zeros((30, 40))
            array = biggus.ArrayAdapter(ndarray, keys=key)
            self.assertEqual(array.shape, shape)

    def test_getitem(self):
        tests = [
            [(30, 40), None, None, (30, 40)],
            [(30, 40), None, (5,), (40,)],
            [(30, 40), None, (slice(None, None), 6), (30,)],
            [(30, 40), None, (slice(None, None), slice(1, 5)), (30, 4)],
            # TODO: Test once implemented
            #[(30, 40), (5,), (slice(None, None),), (40,)],
        ]
        for src_shape, src_keys, keys, result_shape in tests:
            ndarray = numpy.zeros(src_shape)
            array = biggus.ArrayAdapter(ndarray, keys=src_keys)
            result = array.__getitem__(keys)
            self.assertIsInstance(result, biggus.Array)
            self.assertEqual(result.shape, result_shape)

    def test_ndarray(self):
        tests = [
            [(3,), None, [0, 1, 2]],
            [(3,), (1,), [1]],
            [(3,), (slice(None, None, 2),), [0, 2]],
            [(3, 4), None, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]],
            [(3, 4), (1, ), [4, 5, 6, 7]],
            [(3, 4), (1, 3), 7],
        ]
        for src_shape, src_keys, target in tests:
            size = reduce(lambda x, y: x * y, src_shape)
            ndarray = numpy.arange(size).reshape(src_shape)
            array = biggus.ArrayAdapter(ndarray, keys=src_keys)
            result = array.ndarray()
            self.assertIsInstance(result, numpy.ndarray)
            self.assertEqual(array.dtype, result.dtype)
            self.assertEqual(array.shape, result.shape)
            numpy.testing.assert_array_equal(result, result)


if __name__ == '__main__':
    unittest.main()
