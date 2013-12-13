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


class TestAggregation(unittest.TestCase):
    def _test_aggregation(self, biggus_op, numpy_op, **kwargs):
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
            raw_data = np.linspace(0, 1, num=size).reshape(shape)

            # Check the aggregation operation doesn't actually read any
            # data.
            data = AccessCounter(raw_data)
            array = biggus.NumpyArrayAdapter(data)
            op_array = biggus_op(array, axis=0, **kwargs)
            self.assertIsInstance(op_array, biggus.Array)
            self.assertTrue((data.counts == 0).all())

            # Compute the NumPy aggregation, and then wrap the result as
            # an array so we can apply biggus-style indexing.
            numpy_op_data = numpy_op(raw_data, axis=axis, **kwargs)
            numpy_op_array = biggus.NumpyArrayAdapter(numpy_op_data)

            for keys in cuts:
                # Check slicing doesn't actually read any data.
                op_array = op_array[keys]
                self.assertIsInstance(op_array, biggus.Array)
                self.assertTrue((data.counts == 0).all())
                # Update the NumPy result to match
                numpy_op_array = numpy_op_array[keys]

            # Check resolving `op_array` to a NumPy array only reads
            # each relevant source value once.
            op_result = op_array.ndarray()
            self.assertTrue((data.counts <= 1).all())

            # Check the NumPy and biggus numeric values match.
            numpy_result = numpy_op_array.ndarray()
            np.testing.assert_array_almost_equal(op_result, numpy_result)

    def test_mean(self):
        self._test_aggregation(biggus.mean, np.mean)

    def test_std(self):
        self._test_aggregation(biggus.std, np.std, ddof=0)
        self._test_aggregation(biggus.std, np.std, ddof=1)


if __name__ == '__main__':
    unittest.main()
