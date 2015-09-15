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
"""Unit tests for `biggus._normalise_axis`."""

import unittest

from biggus import _normalise_axis
from biggus.tests import mock


ARRAY = mock.Mock(ndim=9)


class TestNone(unittest.TestCase):
    def test(self):
        self.assertIs(_normalise_axis(None, ARRAY), None)


class TestValid(unittest.TestCase):
    def check(self, argument, expected):
        result = _normalise_axis(argument, ARRAY)
        self.assertEqual(result, expected)
        self.assertIsInstance(result, tuple)

    def test_zero(self):
        self.check(0, (0,))

    def test_one(self):
        self.check(1, (1,))

    def test_single_tuple(self):
        self.check((8,), (8,))

    def test_single_list(self):
        self.check([8], (8,))

    def test_multi_tuple(self):
        self.check((3, 2), (3, 2))

    def test_multi_list(self):
        self.check([4, 1], (4, 1))

    def test_negative(self):
        self.check(-1, (8,))

    def test_negative_tuple(self):
        self.check((-1), (8,))

    def test_mixed(self):
        self.check((-1, 3, -2), (8, 3, 7))


class TestInvalidInt(unittest.TestCase):
    def test_builtin(self):
        with self.assertRaises(TypeError):
            _normalise_axis(open, ARRAY)

    def test_dict(self):
        with self.assertRaises(TypeError):
            _normalise_axis({}, ARRAY)

    def test_float(self):
        with self.assertRaises(TypeError):
            _normalise_axis(2.3, ARRAY)

    def test_str(self):
        with self.assertRaises(TypeError):
            _normalise_axis('23', ARRAY)


class TestInvalidIterable(unittest.TestCase):
    def test_mixed_tuple(self):
        with self.assertRaises(TypeError):
            _normalise_axis((2, '23', 6), ARRAY)

    def test_all_bad_tuple(self):
        with self.assertRaises(TypeError):
            _normalise_axis((2.8, '23', {6: 3}), ARRAY)


if __name__ == '__main__':
    unittest.main()
