# (C) British Crown Copyright 2013 - 2015, Met Office
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

import numpy as np
from numpy.testing import assert_array_equal

import biggus
from biggus.tests import AccessCounter


class TestElementwise(unittest.TestCase):
    def _test_elementwise(self, biggus_op, numpy_op):
        # Sequence of tests, defined as:
        #   1. Original array shape1.
        #   2. Original array shape2
        #   3. Sequence of indexing operations to apply.
        tests = [
            [(10, ), (10, ), []],
            [(30, 40), (30, 40), []],
            [(30, 40), (30, 40), (5,)],
            [(10, 30, 1), (1, 40), []],
            [(2, 3, 1), (1, 4), [slice(1, 2)]],
            [(500, 30, 40), (500, 30, 40), [slice(3, 6)]],
            [(500, 30, 40), (500, 30, 40), [(slice(None), slice(3, 6))]],
        ]
        axis = 0
        ddof = 0
        for shape1, shape2, cuts in tests:
            # Define some test data
            raw_data1 = np.linspace(0.0, 1.0, np.prod(shape1)).reshape(shape1)
            raw_data2 = np.linspace(0.2, 1.2, np.prod(shape2)).reshape(shape2)

            # Check the elementwise operation doesn't actually read any
            # data.
            data1 = AccessCounter(raw_data1)
            data2 = AccessCounter(raw_data2)
            array1 = biggus.NumpyArrayAdapter(data1)
            array2 = biggus.NumpyArrayAdapter(data2)
            op_array = biggus_op(array1, array2)
            self.assertIsInstance(op_array, biggus.Array)
            self.assertTrue((data1.counts == 0).all())
            self.assertTrue((data2.counts == 0).all())

            # Compute the NumPy elementwise operation, and then wrap the
            # result as an array so we can apply biggus-style indexing.
            numpy_op_data = numpy_op(raw_data1, raw_data2)
            numpy_op_array = biggus.NumpyArrayAdapter(numpy_op_data)

            for keys in cuts:
                # Check slicing doesn't actually read any data.
                op_array = op_array[keys]
                self.assertIsInstance(op_array, biggus.Array)
                self.assertTrue((data1.counts == 0).all())
                self.assertTrue((data2.counts == 0).all())
                # Update the NumPy result to match
                numpy_op_array = numpy_op_array[keys]

            # Check the NumPy and biggus numeric values match.
            op_result = op_array.ndarray()
            numpy_result = numpy_op_array.ndarray()
            np.testing.assert_array_equal(op_result, numpy_result)

    def test_add(self):
        self._test_elementwise(biggus.add, np.add)

    def test_sub(self):
        self._test_elementwise(biggus.sub, np.subtract)

    def test_multiply(self):
        self._test_elementwise(biggus.multiply, np.multiply)

    def test_true_divide(self):
        self._test_elementwise(biggus.true_divide, np.true_divide)

    def test_floor_divide(self):
        self._test_elementwise(biggus.floor_divide, np.floor_divide)

    def test_divide(self):
        self._test_elementwise(biggus.divide, np.divide)

    def test_power(self):
        self._test_elementwise(biggus.power, np.power)

    def test_add_integer(self):
        # Check that the ElementWise functionality accepts numpy arrays,
        # and the result is as expected.
        r = biggus.add(np.arange(3) * 2, 5)
        assert_array_equal(r.ndarray(), [5, 7, 9])

    def test_divide_float(self):
        r = biggus.divide(np.arange(3.), 2.)
        assert_array_equal(r.ndarray(), [0., 0.5, 1.])

    def test_true_divide_float(self):
        r = biggus.true_divide(np.arange(3.), 2.)
        assert_array_equal(r.ndarray(), [0., 0.5, 1.])

    def test_add_nd_array(self):
        # Check that the ElementWise functionality accepts numpy arrays,
        # and the result is as expected.
        r = biggus.add(np.arange(4), np.arange(4) * 2)
        self.assertIsInstance(r._array1, biggus.NumpyArrayAdapter)
        self.assertIsInstance(r._array2, biggus.NumpyArrayAdapter)
        assert_array_equal(r.ndarray(), [0, 3, 6, 9])

    def test_add_non_supported_type(self):
        # Check that the ElementWise functionality raises a TypeError
        # if neither an Array or np.ndarray is given
        with self.assertRaises(TypeError):
            biggus.add(range(12), np.arange(12))


if __name__ == '__main__':
    unittest.main()
