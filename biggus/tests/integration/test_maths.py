# (C) British Crown Copyright 2015, Met Office
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
"""Integration tests for maths operations."""

import numpy as np
import numpy.ma as ma
import unittest

import biggus
import biggus.tests.unit._aggregation_test_framework as test_framework


class TestSum(unittest.TestCase):

    def test_sum_float(self):
        a = biggus.ConstantArray((1), dtype=np.float32)
        b = a + 1
        self.assertEqual('float32', b.dtype)

    def test_sum_int(self):
        a = biggus.ConstantArray((1), dtype=np.int32)
        b = a + 1
        self.assertEqual('int32', b.dtype)


if __name__ == '__main__':
    unittest.main()
