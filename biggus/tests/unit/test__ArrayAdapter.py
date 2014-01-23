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
"""Unit tests for `biggus._ArrayAdapter`."""

import unittest

from biggus import _ArrayAdapter


class Test___init__(unittest.TestCase):
    def test_concrete(self):
        class FakeConcrete(object):
            shape = ()

        class FakeAdapter(_ArrayAdapter):
            def _apply_keys(self):
                pass

        concrete = FakeConcrete()
        array = FakeAdapter(concrete)
        self.assertIs(array.concrete, concrete)


if __name__ == '__main__':
    unittest.main()
