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
"""Integration tests for swappable engine."""

import unittest

import biggus
from biggus.tests import mock


class Test(unittest.TestCase):
    def test(self):
        # If we switch evaluation engine, does it get used?
        array = biggus._Aggregation(biggus.ConstantArray(3, 2), None, None,
                                    None, None, {})
        return_value = (mock.sentinel.result,)
        engine = mock.Mock(**{'ndarrays.return_value': return_value})
        with mock.patch('biggus.engine', engine):
            result = array.ndarray()
        self.assertIs(result, mock.sentinel.result)


if __name__ == '__main__':
    unittest.main()
