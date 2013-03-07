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

import numpy

import biggus


class _AccessCounter(object):
    """
    Something that acts like a NumPy ndarray, but which records how
    many times each element has been read.

    """
    def __init__(self, ndarray):
        self._ndarray = ndarray
        self.counts = numpy.zeros(ndarray.shape)

    @property
    def ndim(self):
        return self._ndarray.ndim

    @property
    def shape(self):
        return self._ndarray.shape

    def __array__(self):
        return self._ndarray

    def __getitem__(self, keys):
        self.counts[keys] += 1
        return self._ndarray[keys]

    def unique_counts(self):
        return set(numpy.unique(self.counts))


class TestStd(unittest.TestCase):
    def _compare(self, data, axis=0, ddof=0):
        array = biggus.ArrayAdapter(data)
        biggus_std = biggus.std(array, axis=axis, ddof=ddof)
        numpy_std = numpy.std(data, axis=axis, ddof=ddof)
        numpy.testing.assert_array_almost_equal(numpy_std,
                                                biggus_std.ndarray())

    def test_std_1d(self):
        self._compare(numpy.arange(5, dtype='f4'))
        self._compare(numpy.arange(5, dtype='f4'), ddof=1)

    def test_std_2d(self):
        data = numpy.arange(20, dtype='f4').reshape((4, 5))
        self._compare(data)
        self._compare(data.T)
        self._compare(data, ddof=1)
        self._compare(data.T, ddof=1)

    def test_std(self):
        # Sequence of tests, defined as:
        #   1. Original array shape.
        #   2. Sequence of indexing operations to apply.
        tests = [
            [(30, 40), []],
            [(30, 40), [5]],
            [(500, 30, 40), [slice(3, 6)]],
            [(500, 30, 40), [(slice(None), slice(3, 6))]],
        ]
        axis = 0
        ddof = 0
        for shape, cuts in tests:
            print shape, cuts
            # Define some test data
            size = numpy.prod(shape)
            raw_data = numpy.linspace(0, 1, num=size).reshape(shape)

            # "Compute" the biggus standard deviation
            data = _AccessCounter(raw_data)
            array = biggus.ArrayAdapter(data)
            biggus_std = biggus.std(array, axis=axis, ddof=ddof)

            # Compute the NumPy standard deviation, and then wrap the
            # result as an array so we can apply biggus-style indexing.
            numpy_std_data = numpy.std(raw_data, axis=axis, ddof=ddof)
            numpy_std_array = biggus.ArrayAdapter(numpy_std_data)

            # Check the `std` operation doesn't actually read any data.
            std_array = biggus.std(array, axis=0)
            self.assertIsInstance(std_array, biggus.Array)
            self.assertTrue((data.counts == 0).all())

            for keys in cuts:
                # Check slicing doesn't actually read any data.
                std_array = std_array[keys]
                self.assertIsInstance(std_array, biggus.Array)
                self.assertTrue((data.counts == 0).all())
                # Update the NumPy result to match
                numpy_std_array = numpy_std_array[keys]

            # Check resolving `std_array` to a NumPy array only reads
            # each relevant source value once.
            std = std_array.ndarray()
            self.assertTrue((data.counts <= 1).all())

            # Check the NumPy and biggus numeric values match.
            numpy_std = numpy_std_array.ndarray()
            numpy.testing.assert_array_almost_equal(std, numpy_std)


if __name__ == '__main__':
    unittest.main()
