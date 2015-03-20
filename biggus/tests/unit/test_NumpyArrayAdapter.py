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
"""Unit tests for `biggus.NumpyArrayAdapter`."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from biggus import NumpyArrayAdapter, NewAxesArray


class Test___getitem__(unittest.TestCase):
    def test_newaxis(self):
        # Check we have the NewAxesArray.handle_newaxis decorator.
        array = NumpyArrayAdapter(np.arange(24))
        result = array[np.newaxis, :]
        self.assertIsInstance(result, NewAxesArray)
        self.assertEqual(result.shape, (1, 24))


if __name__ == '__main__':
    unittest.main()
