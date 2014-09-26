# (C) British Crown Copyright 2013 - 2014, Met Office
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
import numpy.ma
import numpy.testing

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
        for shape, cuts in tests:
            # Define some test data
            size = np.prod(shape)
            raw_data = np.linspace(0, 1, num=size).reshape(shape)

            for axis in range(len(shape)):
                # Check the aggregation operation doesn't actually read any
                # data.
                data = AccessCounter(raw_data)
                array = biggus.NumpyArrayAdapter(data)
                op_array = biggus_op(array, axis=axis, **kwargs)
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

    def test_var(self):
        self._test_aggregation(biggus.var, np.var, ddof=0)
        self._test_aggregation(biggus.var, np.var, ddof=1)

    def test_mean_nd_array(self):
        r = biggus.mean(np.arange(12), axis=0)
        self.assertIsInstance(r._array, biggus.NumpyArrayAdapter)
        self.assertEqual(r.ndarray(), 5.5)

    def test_add_non_supported_type(self):
        # Check that the Aggregation raises a TypeError
        # if neither an Array or np.ndarray is given
        msg = "list' object has no attribute 'ndim"
        with self.assertRaisesRegexp(AttributeError, msg):
            biggus.mean(range(10), axis=0)


class TestMdtol(unittest.TestCase):
    def setUp(self):
        self.data = np.ma.arange(12).reshape(3, 4)
        self.data[2, 1:] = np.ma.masked

    def _test_mean_with_mdtol(self, data, axis, numpy_result, mdtol=None):
        data = AccessCounter(data)
        array = biggus.NumpyArrayAdapter(data)

        # Perform aggregation.
        if mdtol is None:
            # Allow testing of default when mdtol is None.
            biggus_aggregation = biggus.mean(array, axis=axis)
        else:
            biggus_aggregation = biggus.mean(array, axis=axis, mdtol=mdtol)

        # Check the aggregation operation doesn't actually read any data.
        self.assertTrue((data.counts == 0).all())

        # Check results.
        biggus_result = biggus_aggregation.masked_array()
        # Check resolving `op_array` to a NumPy array only reads
        # each relevant source value once.
        self.assertTrue((data.counts <= 1).all())
        numpy_mask = np.ma.getmaskarray(numpy_result)
        biggus_mask = np.ma.getmaskarray(biggus_result)
        np.testing.assert_array_equal(biggus_mask, numpy_mask)
        np.testing.assert_array_equal(biggus_result[~biggus_mask].data,
                                      numpy_result[~numpy_mask].data)

    def test_mean_mdtol_default(self):
        axis = 0
        expected = np.ma.mean(self.data, axis)
        self._test_mean_with_mdtol(self.data, axis, expected)

    def test_mean_mdtol_one(self):
        axis = 0
        mdtol = 1
        expected = np.ma.mean(self.data, axis)
        self._test_mean_with_mdtol(self.data, axis, expected, mdtol)

    def test_mean_mdtol_zero(self):
        axis = 0
        mdtol = 0
        expected = np.ma.mean(self.data, axis)
        expected.mask = [False, True, True, True]
        self._test_mean_with_mdtol(self.data, axis, expected, mdtol)

    def test_mean_mdtol_fraction_below_axis_zero(self):
        axis = 0
        mdtol = 0.32
        expected = np.ma.mean(self.data, axis)
        expected.mask = [False, True, True, True]
        self._test_mean_with_mdtol(self.data, axis, expected, mdtol)

    def test_mean_mdtol_fraction_above_axis_zero(self):
        axis = 0
        mdtol = 0.34
        expected = np.ma.mean(self.data, axis)
        self._test_mean_with_mdtol(self.data, axis, expected, mdtol)

    def test_mean_mdtol_fraction_below_axis_one(self):
        axis = 1
        mdtol = 0.74
        expected = np.ma.mean(self.data, axis)
        expected.mask = [False, False, True]
        self._test_mean_with_mdtol(self.data, axis, expected, mdtol)

    def test_mean_mdtol_fraction_above_axis_one(self):
        axis = 1
        mdtol = 0.76
        expected = np.ma.mean(self.data, axis)
        self._test_mean_with_mdtol(self.data, axis, expected, mdtol)


class TestFlow(unittest.TestCase):
    def _test_flow(self, axis):
        data = np.arange(3 * 4 * 5, dtype='f4').reshape(3, 4, 5)
        array = biggus.NumpyArrayAdapter(data)
        mean = biggus.mean(array, axis=axis)
        engine = biggus.AllThreadedEngine()
        chunk_size = biggus.MAX_CHUNK_SIZE
        try:
            # Artificially constrain the chunk size to eight bytes to
            # ensure biggus is stepping across axes in the correct
            # order.
            biggus.MAX_CHUNK_SIZE = 8
            op_result, = engine.ndarrays(mean)
        finally:
            biggus.MAX_CHUNK_SIZE = chunk_size
        np_result = np.mean(data, axis=axis)
        np.testing.assert_array_almost_equal(op_result, np_result)

    def test_axis_0(self):
        self._test_flow(0)

    def test_axis_1(self):
        self._test_flow(1)

    def test_axis_2(self):
        self._test_flow(2)


if __name__ == '__main__':
    unittest.main()
