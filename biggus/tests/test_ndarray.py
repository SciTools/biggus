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


class TestNdarray(unittest.TestCase):
    def assert_counts(self, counts, expected_counts):
        self.assertEqual(np.unique(counts), expected_counts)

    def test_dual_aggregation(self):
        # Check the aggregation operations don't actually read any data.
        shape = (500, 30, 40)
        size = np.prod(shape)
        raw_data = np.linspace(0, 1, num=size).reshape(shape)
        counter = AccessCounter(raw_data)
        array = biggus.NumpyArrayAdapter(counter)
        mean_array = biggus.mean(array, axis=0)
        std_array = biggus.std(array, axis=0)
        self.assertIsInstance(mean_array, biggus.Array)
        self.assertIsInstance(std_array, biggus.Array)
        self.assertTrue((counter.counts == 0).all())

        mean, std_dev = biggus.ndarrays([mean_array, std_array])

        # Was the source data read just once?
        self.assert_counts(counter.counts, [1])

    def test_mean_of_difference(self):
        # MEAN(A - B)
        shape = (500, 30, 40)
        size = np.prod(shape)
        raw_data = np.linspace(0, 1, num=size).reshape(shape)
        data = AccessCounter(raw_data * 3)
        a_array = biggus.NumpyArrayAdapter(data)
        data = AccessCounter(raw_data)
        b_array = biggus.NumpyArrayAdapter(data)

        mean_array = biggus.mean(biggus.sub(a_array, b_array), axis=0)

        mean = mean_array.ndarray()
        np.testing.assert_array_almost_equal(mean,
                                             np.mean(raw_data * 2, axis=0))

    def test_sd_and_mean_of_difference(self):
        # MEAN(A - B) and SD(A - B)
        shape = (500, 30, 40)
        size = np.prod(shape)
        raw_data = np.linspace(0, 1, num=size).reshape(shape)
        a_counter = AccessCounter(raw_data * 3)
        a_array = biggus.NumpyArrayAdapter(a_counter)
        b_counter = AccessCounter(raw_data)
        b_array = biggus.NumpyArrayAdapter(b_counter)

        sub_array = biggus.sub(a_array, b_array)
        mean_array = biggus.mean(sub_array, axis=0)
        std_array = biggus.std(sub_array, axis=0)
        mean, std = biggus.ndarrays([mean_array, std_array])

        # Are the resulting numbers equivalent?
        np.testing.assert_array_almost_equal(mean,
                                             np.mean(raw_data * 2, axis=0))
        np.testing.assert_array_almost_equal(std,
                                             np.std(raw_data * 2, axis=0))
        # Was the source data read just once?
        self.assert_counts(a_counter.counts, [1])
        self.assert_counts(b_counter.counts, [1])

    def test_mean_of_a_and_mean_of_difference(self):
        # MEAN(A) and MEAN(A - B)
        shape = (500, 30, 40)
        size = np.prod(shape)
        raw_data = np.linspace(0, 1, num=size).reshape(shape)
        a_counter = AccessCounter(raw_data * 3)
        a_array = biggus.NumpyArrayAdapter(a_counter)
        b_counter = AccessCounter(raw_data)
        b_array = biggus.NumpyArrayAdapter(b_counter)

        sub_array = biggus.sub(a_array, b_array)
        mean_a_array = biggus.mean(a_array, axis=0)
        mean_sub_array = biggus.mean(sub_array, axis=0)
        mean_a, mean_sub = biggus.ndarrays([mean_a_array, mean_sub_array])

        # Are the resulting numbers equivalent?
        np.testing.assert_array_almost_equal(mean_a,
                                             np.mean(raw_data * 3, axis=0))
        np.testing.assert_array_almost_equal(mean_sub,
                                             np.mean(raw_data * 2, axis=0))

        # Was the source data read the minimal number of times?
        self.assert_counts(a_counter.counts, [1])
        self.assert_counts(b_counter.counts, [1])


if __name__ == '__main__':
    unittest.main()
