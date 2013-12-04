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
from biggus.tests import AccessCounter


class TestNdarray(unittest.TestCase):
    def test_dual_aggregation(self):
        # Check the aggregation operations don't actually read any data.
        shape = (500, 30, 40)
        size = numpy.prod(shape)
        raw_data = numpy.linspace(0, 1, num=size).reshape(shape)
        data = AccessCounter(raw_data)
        array = biggus.NumpyArrayAdapter(data)
        mean_array = biggus.mean(array, axis=0)
        std_array = biggus.std(array, axis=0)
        self.assertIsInstance(mean_array, biggus.Array)
        self.assertIsInstance(std_array, biggus.Array)
        self.assertTrue((data.counts == 0).all())

        mean, std_dev = biggus.ndarrays([mean_array, std_array])
        # The first slice is read twice because both `mean` and `std`
        # use it to to bootstrap their rolling calculations.
        self.assertTrue((data.counts[0] == 2).all())
        self.assertTrue((data.counts[1:] == 1).all())


if __name__ == '__main__':
    unittest.main()
