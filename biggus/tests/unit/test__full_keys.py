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
"""Unit tests for `biggus._full_keys`."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from biggus import _full_keys


class Test__full_keys(unittest.TestCase):
    def nslice(self, n):
        return tuple([slice(None)] * n)

    def assertFullSlice(self, keys, ndim, expected):
        result = _full_keys(keys, ndim)
        assert_array_equal(result, tuple(expected))

        # Now check that this is actually the numpy behaviour.
        a = np.empty(range(7, 7 + ndim), dtype=np.bool)
        msg = 'Full slice produces a different result to numpy.'
        assert_array_equal(a[keys], a[expected], msg)

    def test_identity(self):
        self.assertFullSlice((slice(None), slice(None)), 2,
                             [slice(None), slice(None)])

    def test_expand(self):
        self.assertFullSlice((slice(None), ), 2,
                             [slice(None), slice(None)])

    def test_expand_too_many(self):
        with self.assertRaises(IndexError):
            _full_keys((slice(None), 2, 3), 2)

    def test_lh_ellipsis(self):
        self.assertFullSlice((Ellipsis, ), 2,
                             [slice(None), slice(None)])

    def test_rh_ellipsis(self):
        self.assertFullSlice((1, Ellipsis), 2,
                             [1, slice(None)])

    def test_double_ellipsis(self):
        self.assertFullSlice((1, Ellipsis, 1, Ellipsis), 4,
                             [1, slice(None), 1, slice(None)])

    def test_double_ellipsis_new_axis(self):
        self.assertFullSlice((1, Ellipsis, 1, np.newaxis, Ellipsis), 4,
                             [1, slice(None), 1, None, slice(None)])

    def test_new_axis(self):
        self.assertFullSlice((np.newaxis, ), 2,
                             [np.newaxis, slice(None), slice(None)])

    def test_numpy_and_ellipsis_1d(self):
        # The key is that Ellipsis needs to be stripped off the keys.
        self.assertFullSlice((np.array([0, 4, 5, 2]), Ellipsis), 1,
                             [np.array([0, 4, 5, 2])])

    def test_new_axis_lh_and_rh(self):
        self.assertFullSlice((np.newaxis, Ellipsis, None, np.newaxis), 2,
                             [None, slice(None), slice(None), None, None])

    def test_redundant_ellipsis(self):
        keys = (slice(None), Ellipsis, 0, Ellipsis, slice(None))
        self.assertFullSlice(keys, 4,
                             (slice(None), 0, slice(None), slice(None)))

    def test_ellipsis_expands_to_nothing(self):
        keys = (slice(None, None, -1), Ellipsis, slice(1, 2))
        self.assertFullSlice(keys, 2,
                             (slice(None, None, -1), slice(1, 2)))


if __name__ == '__main__':
    unittest.main()
