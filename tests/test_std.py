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


class TestStd(unittest.TestCase):
    def _compare(self, data, axis=0, ddof=0):
        array = biggus.ArrayAdapter(data)
        biggus_std = biggus.std(array, axis=axis, ddof=ddof)
        numpy_std = numpy.std(data, axis=axis, ddof=ddof)
        numpy.testing.assert_array_equal(numpy_std, biggus_std)

    def test_std_1d(self):
        self._compare(numpy.arange(5, dtype='f4'))
        self._compare(numpy.arange(5, dtype='f4'), ddof=1)

    def test_std_2d(self):
        data = numpy.arange(20, dtype='f4').reshape((4, 5))
        self._compare(data)
        self._compare(data.T)
        self._compare(data, ddof=1)
        self._compare(data.T, ddof=1)


if __name__ == '__main__':
    unittest.main()
