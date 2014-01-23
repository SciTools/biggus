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
"""Unit tests for `biggus.Array`."""

import unittest

from biggus import Array


class Test___hash__(unittest.TestCase):
    def test_unhashable(self):
        class FakeArray(Array):
            @property
            def dtype(self):
                pass

            @property
            def shape(self):
                pass

            def __getitem__(self, keys):
                pass

            def ndarray(self, keys):
                pass

            def masked_array(self, keys):
                pass

        array = FakeArray()
        with self.assertRaises(TypeError):
            hash(array)


if __name__ == '__main__':
    unittest.main()
