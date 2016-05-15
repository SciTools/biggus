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
"""Unit tests for `biggus.size`."""

import unittest

from biggus import size


class FakeArray(object):
    def __init__(self, nbytes):
        self.nbytes = nbytes


class Test(unittest.TestCase):
    def _test(self, shape, dtype, expected):
        if not isinstance(shape, tuple):
            shape = (shape,)
        array = FakeArray(shape, dtype)
        self.assertEqual(str(array), expected)

    def test_bytes(self):
        self.assertEqual(size(FakeArray(8)), '8 B')

    def test_1023(self):
        self.assertEqual(size(FakeArray(1023)), '1023 B')

    def test_1024(self):
        self.assertEqual(size(FakeArray(1024)), '1.00 KiB')

    def test_kib(self):
        self.assertEqual(size(FakeArray(4e4)), '39.06 KiB')

    def test_mib(self):
        self.assertEqual(size(FakeArray(4e6)), '3.81 MiB')

    def test_gib(self):
        self.assertEqual(size(FakeArray(4e9)), '3.73 GiB')

    def test_tib(self):
        self.assertEqual(size(FakeArray(4e12)), '3.64 TiB')

    def test_eib(self):
        self.assertEqual(size(FakeArray(4e15)), '3637.98 TiB')


if __name__ == '__main__':
    unittest.main()
