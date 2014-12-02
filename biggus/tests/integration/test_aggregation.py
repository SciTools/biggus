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
"""Integration tests for aggregations."""

import unittest

import numpy as np

import biggus


class Test(unittest.TestCase):
    def test_var_masked(self):
        data = biggus.ConstantArray((10, 400, 720), dtype=np.float32)
        v = biggus.var(data, axis=1)
        result = v.masked_array()
        self.assertIsInstance(result, np.ma.MaskedArray)
        self.assertEqual(result.shape, (10, 720))
        self.assertTrue(np.all(result == 0))
        self.assertTrue(np.all(result.mask == 0))


if __name__ == '__main__':
    unittest.main()
