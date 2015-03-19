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
"""Unit tests for `biggus._sliced_shape`."""

import unittest

import numpy as np

from biggus.tests import key_gen
from biggus import _sliced_shape


class Test__sliced_shape(unittest.TestCase):
    def assertSliceShape(self, shape, keys, expected, not_numpy=False):
        self.assertEqual(_sliced_shape(shape, keys), expected)
        if not not_numpy:
            np_actual = np.empty(shape)[keys].shape
            # Will only fail if numpy is not doing what the test anticipates.
            self.assertEqual(np_actual, expected)

    def test_all_scalar(self):
        keys = key_gen[0, 2, 1]
        self.assertSliceShape([4, 5, 6], keys, ())

    def test_all_sliced(self):
        keys = key_gen[:, :, :]
        self.assertSliceShape([4, 5, 6], keys, (4, 5, 6))

    def test_negative_indexing(self):
        keys = key_gen[:-1, -1:, ::-2]
        self.assertSliceShape([4, 5, 6], keys, (3, 1, 3))

    def test_crazy_sizes(self):
        keys = key_gen[1:10, 1:15, 11:10]
        self.assertSliceShape([4, 5, 6], keys, (3, 4, 0))

    def test_new_axis(self):
        keys = key_gen[np.newaxis, :, np.newaxis, :, 0]
        self.assertSliceShape([4, 5, 6], keys, (1, 4, 1, 5))

    def test_numpy_array_indexing_single(self):
        keys = key_gen[np.arange(2), :, 0]
        self.assertSliceShape([4, 5, 6], keys, (2, 5))

    def test_numpy_array_indexing_double(self):
        keys = key_gen[np.arange(2), :, np.arange(1, 4)]
        self.assertSliceShape([4, 5, 6], keys, (2, 5, 3), not_numpy=True)

    def test_tuple_indexing(self):
        keys = key_gen[(1, 2), :, 0]
        self.assertSliceShape([4, 5, 6], keys, (2, 5))

    def test_invalid_object_indexing(self):
        keys = key_gen[np.nan]
        msg = 'Invalid indexing object "nan"'
        with self.assertRaisesRegexp(ValueError, msg):
            _sliced_shape([4, 5, 6], keys)

    def test_invalid_object_indexing_float(self):
        # A float is a valid indexing object in numpy.
        keys = key_gen[1.2]
        msg = 'Invalid indexing object "1.2"'
        with self.assertRaisesRegexp(ValueError, msg):
            _sliced_shape([4, 5, 6], keys)

    def test_numpy_bool_indexing(self):
        keys = key_gen[0, :, np.arange(6) > 2]
        # The numpy bool indexing appears completely confused for >1d cases.
        self.assertSliceShape([4, 5, 6], keys, (5, 3), not_numpy=True)

    def test_all_sliced_ellipsis(self):
        keys = key_gen[:, :, :, ...]
        self.assertEqual(_sliced_shape([3, 2, 1], keys),
                         (3, 2, 1))


if __name__ == '__main__':
    unittest.main()
