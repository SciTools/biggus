# (C) British Crown Copyright 2013, Met Office
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

import biggus
from biggus.tests import AccessCounter


class TestElementwise(unittest.TestCase):
    def _test_elementwise(self, biggus_op, numpy_op):
        # Sequence of tests, defined as:
        #   1. Original array shape.
        #   2. Sequence of indexing operations to apply.
        tests = [
            [(10, ), []],
            [(30, 40), []],
            [(30, 40), [5]],
            [(500, 30, 40), [slice(3, 6)]],
            [(500, 30, 40), [(slice(None), slice(3, 6))]],
        ]
        axis = 0
        ddof = 0
        for shape, cuts in tests:
            # Define some test data
            size = np.prod(shape)
            raw_data1 = np.linspace(0.0, 1.0, num=size).reshape(shape)
            raw_data2 = np.linspace(0.2, 1.2, num=size).reshape(shape)

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

    def test_add_nd_array(self):
        # Check that the ElementWise functionality accepts numpy arrays,
        # and the result is as expected.
        r = biggus.add(np.arange(12), np.arange(12))
        self.assertIsInstance(r._array1, biggus.NumpyArrayAdapter)
        self.assertIsInstance(r._array2, biggus.NumpyArrayAdapter)
        self.assertEqual(r.ndarray()[-1], 22)

    def test_add_non_supported_type(self):
        # Check that the ElementWise functionality raises a TypeError
        # if neither an Array or np.ndarray is given
        with self.assertRaises(TypeError):
            biggus.add(range(12), np.arange(12))


if __name__ == '__main__':
    unittest.main()
