# (C) British Crown Copyright 2012, Met Office
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

import biggus


class TestArray(unittest.TestCase):
    class _FakeArray(biggus.Array):
        def __init__(self, shape):
            self._shape = shape
        dtype = property(lambda self: None)
        shape = property(lambda self: self._shape)

        def __getitem__(self, keys):
            return None

        def ndarray(self):
            return None

        def masked_array(self):
            return None

    def test_ndim(self):
        shape_ndims = [[(), 0], [(70, 300, 400), 3], [(6,), 1], [(5, 4), 2]]
        for shape, ndim in shape_ndims:
            array = TestArray._FakeArray(shape)
            self.assertEqual(array.ndim, ndim)


if __name__ == '__main__':
    unittest.main()
