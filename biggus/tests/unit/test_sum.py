# (C) British Crown Copyright 2014, Met Office
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
"""Unit tests for `biggus.sum`."""

import numpy as np
import numpy.ma as ma
import unittest

import biggus
import biggus.tests.unit._aggregation_test_framework as test_framework


class Operator(object):
    @property
    def biggus_operator(self):
        return biggus.sum

    @property
    def numpy_operator(self):
        return np.sum

    @property
    def numpy_masked_operator(self):
        return ma.sum


class TestInvalidAxis(Operator, test_framework.InvalidAxis, unittest.TestCase):
    pass


class TestAggregationDtype(
        Operator, test_framework.AggregationDtype, unittest.TestCase):
    pass


class TestNumpyArrayAdapter(
        Operator, test_framework.NumpyArrayAdapter, unittest.TestCase):
    pass


class TestNumpyArrayAdapterMasked(
        Operator, test_framework.NumpyArrayAdapterMasked, unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
