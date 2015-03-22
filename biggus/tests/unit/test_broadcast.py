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
"""Unit tests for `biggus.broadcast`."""

import unittest

import numpy as np

from biggus import ConstantArray, broadcast


class Test_computecompute_broadcast_shape(unittest.TestCase):
    def test_have_enough_memory(self):
        # Using np.broadcast results in the actual data needing to be
        # realised.
        a = ConstantArray([int(10 ** i) for i in range(4, 12)])
        self.assertEqual(broadcast.compute_broadcast_shape(a.shape,
                                                           a.shape),
                         tuple(int(10 ** i) for i in range(4, 12)))

    def assertBroadcast(self, s1, s2, expected):
        # Assert that the operations are symmetric.
        r1 = broadcast.compute_broadcast_shape(s1, s2)
        r2 = broadcast.compute_broadcast_shape(s2, s1)
        self.assertEqual(r1, expected)
        self.assertEqual(r1, r2)
        actual = np.broadcast(np.empty(s1), np.empty(s2)).shape
        self.assertEqual(expected, actual)

    def test_scalar_broadcasting(self):
        self.assertBroadcast([1, 2, 3], [], (1, 2, 3))

    def test_scalar_scalar(self):
        self.assertBroadcast([], [], ())

    def test_rule1_leading_padding(self):
        self.assertBroadcast((1, 2, 3), [3], (1, 2, 3))

    def test_rule2_filling_in_1s(self):
        self.assertBroadcast([1, 2, 3], [3, 1, 3], (3, 2, 3))

    def test_rule3_value_error(self):
        msg = ('operands could not be broadcast together with shapes '
               '\(2\,3\) \(1\,2\,4\)')
        with self.assertRaisesRegexp(ValueError, msg):
            broadcast.compute_broadcast_shape([2, 3], [1, 2, 4])


if __name__ == '__main__':
    unittest.main()
