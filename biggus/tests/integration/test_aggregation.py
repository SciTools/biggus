# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Integration tests for aggregations."""

import unittest

import numpy as np

import biggus
from biggus.tests import set_chunk_size


class Test(unittest.TestCase):
    def test_var_masked(self):
        data = biggus.ConstantArray((10, 400, 720), dtype=np.float32)
        v = biggus.var(data, axis=1)
        result = v.masked_array()
        self.assertIsInstance(result, np.ma.MaskedArray)
        self.assertEqual(result.shape, (10, 720))
        self.assertTrue(np.all(result == 0))
        self.assertTrue(np.all(result.mask == 0))


class TestAggregation__mixed_dtypes(unittest.TestCase):
    def test(self):
        shape = (3, 5, 10)
        a = biggus.ConstantArray(shape, dtype=np.float32)
        b = biggus.ConstantArray(shape, dtype=np.float64)

        with set_chunk_size(32//8*10-1):
            result = biggus.sum(a * b, axis=0).ndarray()
            self.assertEqual(result.shape, shape[1:])


class Test__slices_with_mathematical_filter(unittest.TestCase):

    def setUp(self):
        self.dtype = np.float32

    def _biggus_filter(self, data, weights):
        # Filter a data array (time, <other dimensions>) using information in
        # weights dictionary.
        #
        # Args:
        #
        # * data:
        #     biggus array of the data to be filtered
        # * weights:
        #     dictionary of absolute record offset : weight

        # Build filter_matrix (time to time' mapping).
        shape = data.shape

        # Build filter matrix as a numpy array and then populate.
        filter_matrix_np = np.zeros((shape[0], shape[0])).astype(self.dtype)

        for offset, value in weights.items():
            filter_matrix_np += np.diag([value] * (shape[0] - offset),
                                        k=offset)
            if offset > 0:
                filter_matrix_np += np.diag([value] * (shape[0] - offset),
                                            k=-offset)

        # Create biggus array for filter matrix, adding in other dimensions.
        for _ in shape[1:]:
            filter_matrix_np = filter_matrix_np[..., np.newaxis]

        filter_matrix_bg_single = biggus.NumpyArrayAdapter(filter_matrix_np)

        # Broadcast to correct shape (time, time', lat, lon).
        filter_matrix_bg = biggus.BroadcastArray(
            filter_matrix_bg_single, {i+2: j for i, j in enumerate(shape[1:])})

        # Broadcast filter to same shape.
        biggus_data_for_filter = biggus.BroadcastArray(data[np.newaxis, ...],
                                                       {0: shape[0]})

        # Multiply two arrays together and sum over second time dimension.
        filtered_data = biggus.sum(biggus_data_for_filter * filter_matrix_bg,
                                   axis=1)

        # Cut off records at start and end of output array where the filter
        # cannot be fully applied.
        filter_halfwidth = len(weights) - 1
        filtered_data = filtered_data[filter_halfwidth:-filter_halfwidth]

        return filtered_data

    def test__biggus_filter(self):
        shape = (1451, 1, 1)

        # Generate dummy data as biggus array.
        numpy_data = np.random.random(shape).astype(self.dtype)
        biggus_data = biggus.NumpyArrayAdapter(numpy_data)

        # Information for filter...
        # Dictionary of weights: key = offset (absolute value), value = weight
        weights = {0: 0.4, 1: 0.2, 2: 0.1}
        # This is equivalent to a weights array of [0.1, 0.2, 0.4, 0.2, 0.1].
        filter_halfwidth = len(weights) - 1

        # Filter data
        filtered_biggus_data = self._biggus_filter(biggus_data, weights)

        # Extract eddy component (original data - filtered data).
        eddy_biggus_data = (biggus_data[filter_halfwidth:-filter_halfwidth] -
                            filtered_biggus_data)

        # Aggregate over time dimension.
        mean_eddy_biggus_data = biggus.mean(eddy_biggus_data, axis=0)

        # Force evaluation.
        mean_eddy_numpy_data = mean_eddy_biggus_data.ndarray()

        # Confirm correct shape.
        self.assertEqual(mean_eddy_numpy_data.shape, shape[1:])


if __name__ == '__main__':
    unittest.main()
